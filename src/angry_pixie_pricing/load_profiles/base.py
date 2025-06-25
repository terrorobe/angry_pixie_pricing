"""Base classes for load profile handling."""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


class LoadProfile(ABC):
    """Abstract base class for all load profile implementations."""
    
    def __init__(self, start_date: datetime, end_date: datetime):
        """Initialize load profile with date range.
        
        Args:
            start_date: Start of the load profile period
            end_date: End of the load profile period
        """
        self.start_date = start_date
        self.end_date = end_date
        self._data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def get_profile(self) -> pd.DataFrame:
        """Get the load profile as a DataFrame.
        
        Returns:
            DataFrame with columns:
            - timestamp: Datetime index (15-minute intervals)
            - power_kw: Power consumption in kW
            - energy_kwh: Energy consumption in kWh for the interval
        """
        pass
    
    @property
    def data(self) -> pd.DataFrame:
        """Lazy-load the profile data."""
        if self._data is None:
            self._data = self.get_profile()
        return self._data
    
    def get_total_consumption(self) -> float:
        """Calculate total energy consumption in kWh."""
        return self.data['energy_kwh'].sum()
    
    def get_peak_power(self) -> Tuple[float, datetime]:
        """Get peak power and when it occurred.
        
        Returns:
            Tuple of (peak_power_kw, timestamp)
        """
        peak_idx = self.data['power_kw'].idxmax()
        peak_power = self.data.loc[peak_idx, 'power_kw']
        return peak_power, peak_idx
    
    def get_load_factor(self) -> float:
        """Calculate load factor (average load / peak load)."""
        avg_load = self.data['power_kw'].mean()
        peak_load = self.data['power_kw'].max()
        return avg_load / peak_load if peak_load > 0 else 0
    
    def resample_to_hourly(self) -> pd.DataFrame:
        """Resample 15-minute data to hourly for price calculations.
        
        Returns:
            DataFrame with hourly power and energy values
        """
        hourly = self.data.resample('h').agg({
            'power_kw': 'mean',
            'energy_kwh': 'sum'
        })
        return hourly
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the load profile."""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_consumption_kwh': self.get_total_consumption(),
            'peak_power_kw': self.get_peak_power()[0],
            'peak_timestamp': self.get_peak_power()[1],
            'average_power_kw': self.data['power_kw'].mean(),
            'min_power_kw': self.data['power_kw'].min(),
            'load_factor': self.get_load_factor(),
            'base_load_kw': self.data['power_kw'].quantile(0.1),
            'median_load_kw': self.data['power_kw'].median()
        }
    
    def filter_by_time_of_day(self, start_hour: int, end_hour: int) -> pd.DataFrame:
        """Filter profile data by time of day.
        
        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
            
        Returns:
            Filtered DataFrame
        """
        mask = (self.data.index.hour >= start_hour) & (self.data.index.hour < end_hour)
        return self.data[mask]
    
    def filter_by_day_of_week(self, weekdays: bool = True) -> pd.DataFrame:
        """Filter profile data by weekday/weekend.
        
        Args:
            weekdays: If True, return weekdays; if False, return weekends
            
        Returns:
            Filtered DataFrame
        """
        if weekdays:
            mask = self.data.index.weekday < 5
        else:
            mask = self.data.index.weekday >= 5
        return self.data[mask]