"""Standard load profile templates for different building types."""

from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd


class ProfileType(Enum):
    """Standard profile types."""

    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    RESIDENTIAL_HEAT_PUMP = "residential_heat_pump"
    COMMERCIAL_RETAIL = "commercial_retail"
    INDUSTRIAL_SHIFT = "industrial_shift"


class ProfileTemplate:
    """Base class for load profile templates."""

    def __init__(self, name: str, base_load_factor: float = 0.3):
        """Initialize profile template.

        Args:
            name: Template name
            base_load_factor: Base load as fraction of peak (0-1)
        """
        self.name = name
        self.base_load_factor = base_load_factor
        self._hourly_factors: dict[int, float] = {}
        self._weekday_factor = 1.0
        self._weekend_factor = 0.8
        self._seasonal_factors = {"winter": 1.2, "spring": 0.9, "summer": 0.8, "autumn": 1.0}

    def set_hourly_pattern(self, pattern: dict[int, float]):
        """Set hourly load factors (0-23 hours)."""
        self._hourly_factors = pattern

    def set_weekly_pattern(self, weekday: float, weekend: float):
        """Set weekday/weekend factors."""
        self._weekday_factor = weekday
        self._weekend_factor = weekend

    def set_seasonal_pattern(self, factors: dict[str, float]):
        """Set seasonal adjustment factors."""
        self._seasonal_factors.update(factors)

    def get_factor(self, timestamp: datetime) -> float:
        """Get load factor for a specific timestamp."""
        # Base factor from hour of day
        hour_factor = self._hourly_factors.get(timestamp.hour, 1.0)

        # Weekday/weekend adjustment
        day_factor = self._weekday_factor if timestamp.weekday() < 5 else self._weekend_factor

        # Seasonal adjustment
        month = timestamp.month
        if month in [12, 1, 2]:
            season_factor = self._seasonal_factors["winter"]
        elif month in [3, 4, 5]:
            season_factor = self._seasonal_factors["spring"]
        elif month in [6, 7, 8]:
            season_factor = self._seasonal_factors["summer"]
        else:
            season_factor = self._seasonal_factors["autumn"]

        # Combine factors
        total_factor = hour_factor * day_factor * season_factor

        # Ensure minimum base load
        return max(total_factor, self.base_load_factor)

    def generate_profile(
        self,
        start_date: datetime,
        end_date: datetime,
        peak_power: float,
        total_consumption: float | None = None,
    ) -> pd.DataFrame:
        """Generate load profile based on template.

        Args:
            start_date: Start date
            end_date: End date
            peak_power: Peak power in kW
            total_consumption: Optional total consumption to match (kWh)

        Returns:
            DataFrame with 15-minute interval load data
        """
        # Generate 15-minute intervals
        timestamps = pd.date_range(start=start_date, end=end_date, freq="15min")

        # Calculate factors for each timestamp
        factors = [self.get_factor(ts) for ts in timestamps]

        # Generate power values
        power_values = peak_power * np.array(factors)

        # If total consumption specified, scale to match
        if total_consumption is not None:
            # Calculate current total
            current_total = sum(power_values) * 0.25  # 15 min = 0.25 hours

            # Scale to match target
            if current_total > 0:
                scale_factor = total_consumption / current_total
                power_values *= scale_factor

        # Create DataFrame
        df = pd.DataFrame(
            {
                "power_kw": power_values,
                "energy_kwh": power_values * 0.25,  # 15 min intervals
            },
            index=timestamps,
        )

        return df


class ResidentialProfile(ProfileTemplate):
    """Typical residential load profile."""

    def __init__(self):
        super().__init__("Residential", base_load_factor=0.2)

        # Typical residential pattern
        self.set_hourly_pattern(
            {
                0: 0.3,
                1: 0.25,
                2: 0.2,
                3: 0.2,
                4: 0.2,
                5: 0.25,
                6: 0.4,
                7: 0.6,
                8: 0.7,
                9: 0.5,
                10: 0.4,
                11: 0.4,
                12: 0.5,
                13: 0.5,
                14: 0.4,
                15: 0.4,
                16: 0.5,
                17: 0.7,
                18: 0.9,
                19: 1.0,
                20: 0.9,
                21: 0.7,
                22: 0.5,
                23: 0.4,
            }
        )

        self.set_weekly_pattern(weekday=1.0, weekend=1.1)

        self.set_seasonal_pattern({"winter": 1.3, "spring": 0.9, "summer": 0.7, "autumn": 1.0})


class CommercialProfile(ProfileTemplate):
    """Typical commercial/office load profile."""

    def __init__(self):
        super().__init__("Commercial", base_load_factor=0.3)

        # Typical office pattern
        self.set_hourly_pattern(
            {
                0: 0.3,
                1: 0.3,
                2: 0.3,
                3: 0.3,
                4: 0.3,
                5: 0.3,
                6: 0.4,
                7: 0.6,
                8: 0.8,
                9: 0.9,
                10: 1.0,
                11: 1.0,
                12: 0.9,
                13: 0.95,
                14: 1.0,
                15: 1.0,
                16: 0.95,
                17: 0.8,
                18: 0.6,
                19: 0.5,
                20: 0.4,
                21: 0.35,
                22: 0.3,
                23: 0.3,
            }
        )

        self.set_weekly_pattern(weekday=1.0, weekend=0.3)

        self.set_seasonal_pattern(
            {
                "winter": 1.1,
                "spring": 0.95,
                "summer": 1.2,  # AC load
                "autumn": 0.95,
            }
        )


class IndustrialProfile(ProfileTemplate):
    """Typical industrial load profile (single shift)."""

    def __init__(self):
        super().__init__("Industrial", base_load_factor=0.4)

        # Single shift pattern
        self.set_hourly_pattern(
            {
                0: 0.4,
                1: 0.4,
                2: 0.4,
                3: 0.4,
                4: 0.4,
                5: 0.5,
                6: 0.7,
                7: 0.9,
                8: 1.0,
                9: 1.0,
                10: 1.0,
                11: 1.0,
                12: 0.8,
                13: 1.0,
                14: 1.0,
                15: 1.0,
                16: 0.9,
                17: 0.7,
                18: 0.5,
                19: 0.4,
                20: 0.4,
                21: 0.4,
                22: 0.4,
                23: 0.4,
            }
        )

        self.set_weekly_pattern(weekday=1.0, weekend=0.4)

        self.set_seasonal_pattern({"winter": 1.05, "spring": 1.0, "summer": 0.95, "autumn": 1.0})


def get_standard_profile(profile_type: ProfileType) -> ProfileTemplate:
    """Get a standard profile template by type."""
    profiles = {
        ProfileType.RESIDENTIAL: ResidentialProfile(),
        ProfileType.COMMERCIAL: CommercialProfile(),
        ProfileType.INDUSTRIAL: IndustrialProfile(),
    }

    return profiles.get(profile_type, ResidentialProfile())
