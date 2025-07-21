"""Specialized load profile templates for hospitality sector."""

from datetime import datetime
from enum import Enum

from .templates import ProfileTemplate


class HotelType(Enum):
    """Hotel profile types."""

    KIDS_HOTEL = "kids_hotel"
    BUSINESS_HOTEL = "business_hotel"
    RESORT_HOTEL = "resort_hotel"
    BOUTIQUE_HOTEL = "boutique_hotel"


class HotelFacilityMixin:
    """Mixin for modeling different hotel facility loads."""

    def get_kitchen_load_factor(self, hour: int, occupancy: float = 0.7) -> float:
        """Restaurant/kitchen load pattern with realistic prep-before-service peaks.

        Commercial kitchens ramp up 1-2 hours before meal service for equipment
        preheat, prep work, and cleaning. Peak power is during prep, not service.

        Breakfast served 7:00-9:30, Lunch 12:00-14:00, Dinner 18:00-21:00
        """
        factors = {
            # Early morning prep and breakfast
            5: 0.3,  # Early prep starts
            6: 0.7,  # Breakfast prep ramp-up (ovens, grills preheat)
            7: 0.9,  # Breakfast prep peak (pre-service)
            8: 0.8,  # Breakfast service
            9: 0.6,  # Late breakfast, cleanup starts
            10: 0.4,  # Breakfast cleanup
            # Lunch prep and service
            11: 0.8,  # Lunch prep ramp-up
            12: 0.9,  # Lunch prep peak (pre-service)
            13: 0.8,  # Lunch service
            14: 0.6,  # Late lunch, cleanup
            15: 0.3,  # Lunch cleanup, prep break
            # Dinner prep and service
            16: 0.6,  # Dinner prep starts
            17: 1.0,  # Dinner prep peak (pre-service, highest load)
            18: 0.9,  # Dinner service starts
            19: 0.8,  # Peak dinner service
            20: 0.7,  # Late dinner service
            21: 0.5,  # Dinner cleanup
            22: 0.3,  # Final cleanup
            23: 0.2,  # Late cleanup
        }
        base_load = 0.2  # Refrigeration, freezers always running
        return base_load + factors.get(hour, 0.1) * occupancy

    def get_wellness_load_factor(self, hour: int, is_kids_hotel: bool = True) -> float:
        """Wellness area (pool, spa) load pattern.

        Kids hotels typically have earlier pool hours.
        """
        if is_kids_hotel:
            # Kids pool hours: 8-20
            if 8 <= hour < 20:
                peak_hours = [10, 11, 15, 16, 17]  # Morning and afternoon peaks
                return 0.9 if hour in peak_hours else 0.7
            return 0.3  # Filtration, heating minimum
        # Adult wellness: 6-22
        if 6 <= hour < 22:
            return 0.8
        return 0.3

    def get_activity_area_load(self, hour: int) -> float:
        """Activity area load (game rooms, entertainment).

        Kids activities peak after meals and in bad weather.
        """
        factors = {
            9: 0.5,
            10: 0.7,
            11: 0.8,  # Morning activities
            14: 0.6,
            15: 0.8,
            16: 0.9,
            17: 0.9,  # Afternoon peak
            18: 0.7,
            19: 0.8,
            20: 0.6,  # Evening activities
        }
        return factors.get(hour, 0.1)

    def get_guest_room_load(self, hour: int, occupancy: float = 0.7) -> float:
        """Guest room load pattern (HVAC, lighting, devices)."""
        # Base HVAC load
        base = 0.3

        # Room usage patterns
        if 6 <= hour < 9:  # Morning routine
            room_factor = 0.8
        elif 9 <= hour < 17:  # Mostly empty
            room_factor = 0.2
        elif 17 <= hour < 23:  # Evening in rooms
            room_factor = 0.9
        else:  # Night
            room_factor = 0.6

        return base + room_factor * occupancy * 0.5

    def get_common_area_load(self, hour: int) -> float:
        """Common areas (lobby, corridors, reception)."""
        if 7 <= hour < 22:
            return 0.8  # Full lighting, HVAC
        return 0.4  # Reduced lighting, security


class KidsHotelProfile(ProfileTemplate, HotelFacilityMixin):
    """Load profile for kids-focused hotel with specialized facilities."""

    def __init__(
        self, occupancy_rate: float = 0.7, facility_weights: dict[str, float] | None = None,
    ):
        """Initialize kids hotel profile.

        Args:
            occupancy_rate: Average occupancy (0-1)
            facility_weights: Relative power draw of facilities
                - kitchen: Restaurant/kitchen systems
                - wellness: Pool, spa, water features
                - activity: Game rooms, entertainment
                - rooms: Guest rooms
                - common: Common areas
        """
        super().__init__("Kids Hotel", base_load_factor=0.35)

        self.occupancy_rate = occupancy_rate
        self.facility_weights = facility_weights or {
            "kitchen": 0.25,
            "wellness": 0.20,
            "activity": 0.10,
            "rooms": 0.30,
            "common": 0.15,
        }

        # Kids hotels have different seasonal patterns
        self.set_seasonal_pattern(
            {
                "winter": 0.9,  # Indoor activities
                "spring": 1.0,  # School holidays
                "summer": 1.2,  # Peak season, AC load
                "autumn": 0.95,
            },
        )

        # Different weekend pattern (higher occupancy)
        self.set_weekly_pattern(weekday=0.9, weekend=1.1)

    def get_factor(self, timestamp: datetime) -> float:
        """Calculate combined load factor for timestamp."""
        hour = timestamp.hour

        # Calculate facility-specific loads
        kitchen_load = self.get_kitchen_load_factor(hour, self.occupancy_rate)
        wellness_load = self.get_wellness_load_factor(hour, is_kids_hotel=True)
        activity_load = self.get_activity_area_load(hour)
        room_load = self.get_guest_room_load(hour, self.occupancy_rate)
        common_load = self.get_common_area_load(hour)

        # Weighted combination
        combined_load = (
            kitchen_load * self.facility_weights["kitchen"]
            + wellness_load * self.facility_weights["wellness"]
            + activity_load * self.facility_weights["activity"]
            + room_load * self.facility_weights["rooms"]
            + common_load * self.facility_weights["common"]
        )

        # Apply weekly and seasonal adjustments
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

        return max(combined_load * day_factor * season_factor, self.base_load_factor)


class DayNightSplitProfile(ProfileTemplate):
    """Profile based on day/night consumption split."""

    def __init__(
        self,
        name: str,
        day_consumption_ratio: float,
        night_consumption_ratio: float,
        base_profile: ProfileTemplate | None = None,
    ):
        """Initialize split profile.

        Args:
            name: Profile name
            day_consumption_ratio: Fraction of total consumption 06-22
            night_consumption_ratio: Fraction of total consumption 22-06
            base_profile: Base profile to modify
        """
        super().__init__(name, base_load_factor=0.3)

        self.day_ratio = day_consumption_ratio
        self.night_ratio = night_consumption_ratio
        self.base_profile = base_profile or KidsHotelProfile()

        # Validate ratios
        if abs((day_consumption_ratio + night_consumption_ratio) - 1.0) > 0.01:
            msg = "Day and night ratios must sum to 1.0"
            raise ValueError(msg)

    def get_factor(self, timestamp: datetime) -> float:
        """Get load factor adjusted for day/night split."""
        hour = timestamp.hour

        # Get base profile factor
        base_factor = self.base_profile.get_factor(timestamp)

        # Calculate expected ratios from base profile
        # This is a simplified calculation - in practice would integrate over full period
        is_day_hour = 6 <= hour < 22
        day_hours = 16
        night_hours = 8

        if is_day_hour:
            # Scale up or down to match day consumption ratio
            # Assuming uniform distribution as starting point
            expected_day_ratio = day_hours / 24
            adjustment = self.day_ratio / expected_day_ratio
        else:
            # Scale for night hours
            expected_night_ratio = night_hours / 24
            adjustment = self.night_ratio / expected_night_ratio

        return base_factor * adjustment
