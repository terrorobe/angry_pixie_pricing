"""Reconstruct load profiles from billing data."""

from datetime import datetime
from typing import Any

import pandas as pd

from .base import LoadProfile
from .templates import ProfileTemplate, ProfileType, get_standard_profile


class BillingReconstructedProfile(LoadProfile):
    """Load profile reconstructed from billing data constraints."""

    def __init__(
        self,
        start_date: datetime,
        end_date: datetime,
        total_consumption: float,
        peak_power: float,
        peak_window_minutes: int = 15,
        profile_template: ProfileTemplate | None = None,
        profile_type: ProfileType | None = None,
        custom_constraints: dict[str, Any] | None = None,
    ):
        """Initialize billing-based profile reconstruction.

        Args:
            start_date: Billing period start
            end_date: Billing period end
            total_consumption: Total energy consumption in kWh
            peak_power: Measured peak power in kW
            peak_window_minutes: Duration of peak measurement window (default 15)
            profile_template: Custom profile template to use
            profile_type: Standard profile type to use
            custom_constraints: Additional constraints for reconstruction
                - known_peak_time: When peak occurred (datetime or hour)
                - min_base_load: Minimum base load in kW
                - operating_hours: Dict with start/end hours per weekday
                - holiday_dates: List of dates with reduced load
                - day_night_split: Tuple of (day_kwh, night_kwh) consumption
                - day_night_ratio: Tuple of (day_ratio, night_ratio) if absolute values not known
        """
        super().__init__(start_date, end_date)

        self.total_consumption = total_consumption
        self.peak_power = peak_power
        self.peak_window_minutes = peak_window_minutes

        # Get profile template
        if profile_template:
            self.template = profile_template
        elif profile_type:
            self.template = get_standard_profile(profile_type)
        else:
            # Default to residential
            self.template = get_standard_profile(ProfileType.RESIDENTIAL)

        # Parse custom constraints
        self.constraints = custom_constraints or {}

    def get_profile(self) -> pd.DataFrame:
        """Reconstruct load profile from billing constraints."""
        # Generate initial profile from template
        df = self.template.generate_profile(
            self.start_date,
            self.end_date,
            self.peak_power,
            self.total_consumption,
        )

        # Apply custom constraints
        df = self._apply_constraints(df)

        # Ensure peak power constraint is met
        df = self._enforce_peak_constraint(df)

        # Final adjustment to match total consumption exactly
        return self._adjust_total_consumption(df)

    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply custom constraints to the profile."""
        # Day/night split constraint (applied first as it's a major adjustment)
        if "day_night_split" in self.constraints or "day_night_ratio" in self.constraints:
            df = self._apply_day_night_constraint(df)

        # Known peak time
        if "known_peak_time" in self.constraints:
            df = self._apply_peak_time_constraint(df, self.constraints["known_peak_time"])

        # Minimum base load
        if "min_base_load" in self.constraints:
            min_load = self.constraints["min_base_load"]
            df["power_kw"] = df["power_kw"].clip(lower=min_load)

        # Operating hours
        if "operating_hours" in self.constraints:
            df = self._apply_operating_hours(df, self.constraints["operating_hours"])

        # Holiday adjustments
        if "holiday_dates" in self.constraints:
            df = self._apply_holiday_adjustments(df, self.constraints["holiday_dates"])

        # Recalculate energy after power adjustments
        df["energy_kwh"] = df["power_kw"] * 0.25

        return df

    def _apply_day_night_constraint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply day/night consumption split constraint."""
        # Define day (06:00-22:00) and night (22:00-06:00) masks
        day_mask = (df.index.hour >= 6) & (df.index.hour < 22)
        night_mask = ~day_mask

        # Get current consumption split
        current_day_kwh = df.loc[day_mask, "energy_kwh"].sum()
        current_night_kwh = df.loc[night_mask, "energy_kwh"].sum()
        current_day_kwh + current_night_kwh

        # Determine target split
        if "day_night_split" in self.constraints:
            # Absolute values provided
            target_day_kwh, target_night_kwh = self.constraints["day_night_split"]
            # Scale to match total consumption
            split_total = target_day_kwh + target_night_kwh
            scale_factor = self.total_consumption / split_total
            target_day_kwh *= scale_factor
            target_night_kwh *= scale_factor
        else:
            # Ratios provided
            day_ratio, night_ratio = self.constraints["day_night_ratio"]
            target_day_kwh = self.total_consumption * day_ratio
            target_night_kwh = self.total_consumption * night_ratio

        # Calculate adjustment factors
        day_adjustment = target_day_kwh / current_day_kwh if current_day_kwh > 0 else 1.0

        night_adjustment = target_night_kwh / current_night_kwh if current_night_kwh > 0 else 1.0

        # Apply adjustments
        df.loc[day_mask, "power_kw"] *= day_adjustment
        df.loc[night_mask, "power_kw"] *= night_adjustment

        # Recalculate energy
        df["energy_kwh"] = df["power_kw"] * 0.25

        return df

    def _apply_peak_time_constraint(self, df: pd.DataFrame, peak_time: Any) -> pd.DataFrame:
        """Apply known peak time constraint."""
        if isinstance(peak_time, datetime):
            # Specific datetime provided
            peak_idx = df.index.get_indexer([peak_time], method="nearest")[0]
            current_peak_idx = df["power_kw"].idxmax()

            # Swap values if needed
            if peak_idx != current_peak_idx:
                peak_value = df["power_kw"].max()
                old_value = df.iloc[peak_idx]["power_kw"]

                df.loc[df.index[peak_idx], "power_kw"] = peak_value
                df.loc[current_peak_idx, "power_kw"] = old_value

        elif isinstance(peak_time, int):
            # Hour of day provided
            hour_mask = df.index.hour == peak_time
            if hour_mask.any():
                # Find highest value in that hour across all days
                hour_data = df[hour_mask]
                max_hour_idx = hour_data["power_kw"].idxmax()

                # Ensure this is the global peak
                current_max = df["power_kw"].max()
                df.loc[max_hour_idx, "power_kw"] = current_max

        return df

    def _apply_operating_hours(self, df: pd.DataFrame, operating_hours: dict[str | int, Any]) -> pd.DataFrame:
        """Apply operating hours constraints."""
        for day, hours in operating_hours.items():
            day_mask = df.index.weekday == day if isinstance(day, int) else df.index.day_name() == day

            # Outside operating hours
            before_mask = day_mask & (df.index.hour < hours.get("start", 0))
            after_mask = day_mask & (df.index.hour >= hours.get("end", 24))

            # Reduce load outside operating hours
            reduction_factor = hours.get("off_hours_factor", 0.3)
            df.loc[before_mask | after_mask, "power_kw"] *= reduction_factor

        return df

    def _apply_holiday_adjustments(
        self,
        df: pd.DataFrame,
        holiday_dates: list[datetime],
    ) -> pd.DataFrame:
        """Apply holiday load reductions."""
        for holiday in holiday_dates:
            holiday_mask = df.index.date == holiday.date()
            df.loc[holiday_mask, "power_kw"] *= 0.5  # 50% reduction on holidays

        return df

    def _enforce_peak_constraint(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the peak power constraint is exactly met."""
        current_peak = df["power_kw"].max()

        if current_peak != self.peak_power:
            # Scale to match peak
            scale_factor = self.peak_power / current_peak
            df["power_kw"] *= scale_factor
            df["energy_kwh"] = df["power_kw"] * 0.25

        return df

    def _adjust_total_consumption(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final adjustment to match total consumption exactly."""
        current_total = df["energy_kwh"].sum()

        if current_total != self.total_consumption:
            # Scale uniformly to match
            scale_factor = self.total_consumption / current_total
            df["power_kw"] *= scale_factor
            df["energy_kwh"] *= scale_factor

        return df

    def validate_reconstruction(self) -> dict[str, Any]:
        """Validate the reconstructed profile against constraints."""
        df = self.data

        actual_total = df["energy_kwh"].sum()
        actual_peak = df["power_kw"].max()

        return {
            "total_consumption_target": self.total_consumption,
            "total_consumption_actual": actual_total,
            "total_consumption_error": abs(actual_total - self.total_consumption),
            "peak_power_target": self.peak_power,
            "peak_power_actual": actual_peak,
            "peak_power_error": abs(actual_peak - self.peak_power),
            "load_factor": df["power_kw"].mean() / actual_peak,
            "base_load": df["power_kw"].quantile(0.1),
            "constraints_applied": list(self.constraints.keys()),
            "template_used": self.template.name,
        }

    @classmethod
    def from_billing_data(
        cls,
        billing_dict: dict[str, Any],
        profile_type: ProfileType | None = None,
    ) -> "BillingReconstructedProfile":
        """Create profile from billing data dictionary.

        Args:
            billing_dict: Dictionary with billing data:
                - start_date: Billing period start
                - end_date: Billing period end
                - total_kwh: Total consumption
                - peak_kw: Peak power
                - building_type: Optional building type hint
            profile_type: Override profile type

        Returns:
            BillingReconstructedProfile instance
        """
        # Determine profile type
        if not profile_type and "building_type" in billing_dict:
            building_map = {
                "residential": ProfileType.RESIDENTIAL,
                "commercial": ProfileType.COMMERCIAL,
                "industrial": ProfileType.INDUSTRIAL,
                "office": ProfileType.COMMERCIAL,
                "retail": ProfileType.COMMERCIAL_RETAIL,
                "factory": ProfileType.INDUSTRIAL,
            }
            profile_type = building_map.get(
                billing_dict["building_type"].lower(),
                ProfileType.RESIDENTIAL,
            )

        return cls(
            start_date=billing_dict["start_date"],
            end_date=billing_dict["end_date"],
            total_consumption=billing_dict["total_kwh"],
            peak_power=billing_dict["peak_kw"],
            profile_type=profile_type,
        )
