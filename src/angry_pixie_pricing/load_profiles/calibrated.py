"""Calibrated load profile that can be applied to any date range."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import LoadProfile
from .templates import ProfileTemplate


class CalibratedProfile(LoadProfile):
    """Load profile calibrated from billing data that can be applied to any date range."""

    def __init__(
        self,
        calculation_start_date: datetime,
        calculation_end_date: datetime,
        daily_kwh: float,
        peak_kw: float,
        profile_template: ProfileTemplate,
        day_night_split: tuple | None = None,
        custom_constraints: dict[str, Any] | None = None,
    ):
        """Initialize calibrated profile.

        Args:
            calculation_start_date: Start date for cost calculation
            calculation_end_date: End date for cost calculation
            daily_kwh: Average daily consumption in kWh (from billing data)
            peak_kw: Peak power in kW (from billing data)
            profile_template: Profile template to use
            day_night_split: Tuple of (daily_day_kwh, daily_night_kwh)
            custom_constraints: Additional constraints
        """
        super().__init__(calculation_start_date, calculation_end_date)

        self.daily_kwh = daily_kwh
        self.peak_kw = peak_kw
        self.profile_template = profile_template
        self.day_night_split = day_night_split
        self.custom_constraints = custom_constraints or {}

        # Calculate total consumption for the calculation period
        calculation_days = (calculation_end_date - calculation_start_date).days + 1
        self.total_consumption = daily_kwh * calculation_days

    def get_profile(self) -> pd.DataFrame:
        """Generate load profile for the calculation period."""
        # Create a single day profile first (as template)
        day_start = datetime(2024, 7, 15, 0, 0)  # Reference day
        day_end = datetime(2024, 7, 15, 23, 59, 59)

        # Generate template day profile
        template_profile = self.profile_template.generate_profile(
            day_start, day_end, self.peak_kw, self.daily_kwh
        )

        # Apply day/night constraints to template if provided
        if self.day_night_split:
            template_profile = self._apply_day_night_split(template_profile)

        # Scale template to match total daily consumption exactly
        current_daily = template_profile["energy_kwh"].sum()
        if current_daily > 0:
            scale_factor = self.daily_kwh / current_daily
            template_profile["power_kw"] *= scale_factor
            template_profile["energy_kwh"] *= scale_factor

        # Now replicate this daily pattern for the entire calculation period
        return self._replicate_daily_pattern(template_profile)

    def _apply_day_night_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply day/night split to template profile."""
        daily_day_kwh, daily_night_kwh = self.day_night_split

        # Define day/night masks
        day_mask = (df.index.hour >= 6) & (df.index.hour < 22)
        night_mask = ~day_mask

        # Get current split
        current_day_kwh = df.loc[day_mask, "energy_kwh"].sum()
        current_night_kwh = df.loc[night_mask, "energy_kwh"].sum()

        # Calculate adjustment factors
        day_adjustment = daily_day_kwh / current_day_kwh if current_day_kwh > 0 else 1.0

        night_adjustment = daily_night_kwh / current_night_kwh if current_night_kwh > 0 else 1.0

        # Apply adjustments
        df.loc[day_mask, "power_kw"] *= day_adjustment
        df.loc[night_mask, "power_kw"] *= night_adjustment

        # Recalculate energy
        df["energy_kwh"] = df["power_kw"] * 0.25

        return df

    def _replicate_daily_pattern(self, template_day: pd.DataFrame) -> pd.DataFrame:
        """Replicate daily pattern across the calculation period."""
        # Generate all timestamps for calculation period
        timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq="15min")

        # Create result DataFrame
        result = pd.DataFrame(index=timestamps)

        # Get template pattern (24 hours * 4 = 96 intervals)
        template_power = template_day["power_kw"].values
        template_energy = template_day["energy_kwh"].values

        # Replicate pattern
        power_values = []
        energy_values = []

        for ts in timestamps:
            # Find corresponding time in template (0-95)
            hour = ts.hour
            minute = ts.minute
            interval_idx = hour * 4 + minute // 15

            # Add some day-to-day variation (±5%)
            variation = np.random.normal(1.0, 0.05)

            # Apply seasonal and weekly factors
            seasonal_factor = self._get_seasonal_factor(ts)
            weekly_factor = self._get_weekly_factor(ts)

            total_factor = variation * seasonal_factor * weekly_factor

            power_values.append(template_power[interval_idx] * total_factor)
            energy_values.append(template_energy[interval_idx] * total_factor)

        result["power_kw"] = power_values
        result["energy_kwh"] = energy_values

        # Final adjustment to ensure total consumption matches target
        actual_total = result["energy_kwh"].sum()
        target_total = self.total_consumption

        if actual_total > 0:
            adjustment = target_total / actual_total
            result["power_kw"] *= adjustment
            result["energy_kwh"] *= adjustment

        return result

    def _get_seasonal_factor(self, timestamp: datetime) -> float:
        """Get seasonal adjustment factor."""
        month = timestamp.month

        # Default seasonal pattern
        seasonal_factors = {
            1: 1.2,
            2: 1.15,
            3: 1.0,
            4: 0.95,
            5: 0.9,
            6: 0.85,
            7: 0.8,
            8: 0.8,
            9: 0.9,
            10: 1.0,
            11: 1.1,
            12: 1.2,
        }

        # Kids hotel specific: higher summer load (pool, AC)
        if hasattr(self.profile_template, "facility_weights"):
            if self.profile_template.facility_weights.get("wellness", 0) > 0.2:
                # Pool-heavy hotel - summer peak
                seasonal_factors.update({6: 1.1, 7: 1.2, 8: 1.2, 9: 1.0})

        return seasonal_factors.get(month, 1.0)

    def _get_weekly_factor(self, timestamp: datetime) -> float:
        """Get weekly adjustment factor."""
        if timestamp.weekday() < 5:  # Weekday
            return 1.0
        else:  # Weekend
            # Hotels typically busier on weekends
            return 1.1

    def get_daily_pattern(self) -> pd.DataFrame:
        """Get the calibrated daily pattern (for visualization)."""
        # Return first day of the profile
        daily_data = self.data.iloc[:96]  # First 24 hours
        return daily_data

    def get_calibration_info(self) -> dict[str, Any]:
        """Get information about the calibration."""
        calc_days = (self.end_date - self.start_date).days + 1

        info = {
            "daily_kwh_target": self.daily_kwh,
            "peak_kw_target": self.peak_kw,
            "calculation_period_days": calc_days,
            "total_consumption_target": self.total_consumption,
            "template_used": self.profile_template.name
            if hasattr(self.profile_template, "name")
            else "Unknown",
        }

        if self.day_night_split:
            daily_day, daily_night = self.day_night_split
            info.update(
                {
                    "daily_day_kwh": daily_day,
                    "daily_night_kwh": daily_night,
                    "day_night_ratio": f"{daily_day / (daily_day + daily_night) * 100:.1f}% / {daily_night / (daily_day + daily_night) * 100:.1f}%",
                }
            )

        return info
