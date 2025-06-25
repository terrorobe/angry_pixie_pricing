"""Load profile handling for electricity consumption analysis."""

from .base import LoadProfile
from .smart_meter import SmartMeterProfile
from .billing import BillingReconstructedProfile
from .templates import (
    ResidentialProfile,
    CommercialProfile,
    IndustrialProfile,
    ProfileTemplate,
    ProfileType,
    get_standard_profile
)
from .hotel_profiles import (
    KidsHotelProfile,
    HotelType,
    DayNightSplitProfile
)

__all__ = [
    'LoadProfile',
    'SmartMeterProfile', 
    'BillingReconstructedProfile',
    'ResidentialProfile',
    'CommercialProfile',
    'IndustrialProfile',
    'ProfileTemplate',
    'ProfileType',
    'get_standard_profile',
    'KidsHotelProfile',
    'HotelType',
    'DayNightSplitProfile'
]