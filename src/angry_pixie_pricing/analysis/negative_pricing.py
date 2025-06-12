"""Analysis of negative and near-zero electricity pricing patterns.

This module analyzes periods when electricity prices are negative or very low,
typically driven by renewable energy oversupply (especially solar PV).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import requests
from dataclasses import dataclass


@dataclass
class NegativePricingMetrics:
    """Container for negative pricing analysis results."""
    total_hours: int
    negative_hours: int
    near_zero_hours: int  # Price <= 5 EUR/MWh
    negative_percentage: float
    near_zero_percentage: float
    avg_hours_per_day: float
    max_consecutive_hours: int
    monthly_breakdown: Dict[int, Dict[str, float]]
    hourly_breakdown: Dict[int, Dict[str, float]]


class NegativePricingAnalyzer:
    """Analyzer for negative and near-zero electricity pricing patterns."""
    
    def __init__(self, region: str):
        """
        Initialize analyzer for a specific region.
        
        Args:
            region: Region/country code
        """
        self.region = region.upper()
        
        # Solar potential data (kWh/m²/day) by month for European regions
        # APPROXIMATE DATA compiled from multiple sources:
        # - EU PVGIS (Photovoltaic Geographical Information System): https://re.jrc.ec.europa.eu/pvg_tools/en/
        # - Global Solar Atlas (World Bank/Solargis): https://globalsolaratlas.info/
        # - Copernicus Climate Data Store: https://climate.copernicus.eu/
        # 
        # NOTE: These are simplified country-level monthly averages for modeling purposes.
        # Actual solar irradiation varies significantly by specific location within countries.
        # For precise analysis, use location-specific PVGIS data.
        self.solar_potential = {
            'DE': [0.7, 1.2, 2.2, 3.6, 4.8, 5.2, 5.0, 4.1, 2.8, 1.6, 0.8, 0.5],  # Germany
            'FR': [1.0, 1.6, 2.8, 4.2, 5.4, 6.0, 6.2, 5.2, 3.6, 2.2, 1.2, 0.8],  # France  
            'ES': [1.8, 2.8, 4.2, 5.8, 7.0, 7.8, 8.2, 7.2, 5.4, 3.6, 2.2, 1.6],  # Spain
            'IT': [1.4, 2.2, 3.6, 5.0, 6.4, 7.2, 7.6, 6.8, 4.8, 3.0, 1.8, 1.2],  # Italy
            'NL': [0.6, 1.0, 2.0, 3.4, 4.6, 5.0, 4.8, 3.8, 2.4, 1.4, 0.7, 0.4],  # Netherlands
            'AT': [1.0, 1.8, 3.0, 4.4, 5.6, 6.0, 6.2, 5.4, 3.8, 2.4, 1.2, 0.8],  # Austria
            'BE': [0.6, 1.1, 2.1, 3.5, 4.7, 5.1, 4.9, 3.9, 2.5, 1.5, 0.8, 0.5],  # Belgium
            'CH': [1.2, 2.0, 3.4, 4.8, 6.0, 6.4, 6.6, 5.8, 4.2, 2.6, 1.4, 1.0],  # Switzerland
            'DK': [0.4, 0.9, 1.9, 3.2, 4.4, 4.8, 4.6, 3.6, 2.2, 1.2, 0.6, 0.3],  # Denmark
            'NO': [0.2, 0.6, 1.4, 2.8, 4.0, 4.4, 4.2, 3.2, 1.8, 0.8, 0.3, 0.1],  # Norway
            'SE': [0.3, 0.8, 1.8, 3.2, 4.6, 5.0, 4.8, 3.6, 2.0, 0.9, 0.4, 0.2],  # Sweden
            'PL': [0.8, 1.4, 2.6, 3.8, 4.8, 5.2, 5.0, 4.2, 2.8, 1.6, 0.9, 0.6],  # Poland
            'CZ': [0.9, 1.5, 2.7, 4.0, 5.0, 5.4, 5.2, 4.4, 3.0, 1.8, 1.0, 0.7],  # Czech Republic
        }
        
        # Current solar capacity estimates (GW) - approximate 2024 values
        # APPROXIMATE DATA from various industry sources:
        # - SolarPower Europe annual reports
        # - IRENA Global Energy Transformation statistics
        # - National renewable energy agencies
        # NOTE: These are rough estimates and change frequently with new installations.
        self.current_solar_capacity = {
            'DE': 80, 'FR': 18, 'ES': 25, 'IT': 26, 'NL': 19, 'AT': 4,
            'BE': 8, 'CH': 5, 'DK': 3, 'NO': 0.3, 'SE': 1.5, 'PL': 15, 'CZ': 3
        }
        
        # Grid flexibility factors (0-1, higher = more flexible)
        # ESTIMATED VALUES based on grid characteristics:
        # - Hydroelectric storage capacity (Norway, Sweden, Switzerland high)
        # - Grid interconnection strength 
        # - Demand response capabilities
        # - Energy storage deployment
        # NOTE: These are simplified estimates for modeling purposes.
        self.grid_flexibility = {
            'DE': 0.7, 'FR': 0.8, 'ES': 0.6, 'IT': 0.5, 'NL': 0.8, 'AT': 0.7,
            'BE': 0.7, 'CH': 0.9, 'DK': 0.9, 'NO': 0.95, 'SE': 0.9, 'PL': 0.5, 'CZ': 0.6
        }
    
    def analyze_negative_pricing_patterns(
        self,
        df: pd.DataFrame,
        near_zero_threshold: float = 5.0
    ) -> NegativePricingMetrics:
        """
        Analyze negative and near-zero pricing patterns in the data.
        
        Args:
            df: DataFrame with columns ['timestamp', 'price', 'unit']
            near_zero_threshold: Price threshold for "near-zero" classification (EUR/MWh)
            
        Returns:
            NegativePricingMetrics object with comprehensive analysis
        """
        if df.empty:
            return self._empty_metrics()
        
        # Basic statistics
        total_hours = len(df)
        negative_mask = df['price'] < 0
        near_zero_mask = df['price'] <= near_zero_threshold
        
        negative_hours = negative_mask.sum()
        near_zero_hours = near_zero_mask.sum()
        
        negative_percentage = (negative_hours / total_hours) * 100
        near_zero_percentage = (near_zero_hours / total_hours) * 100
        
        # Calculate daily averages
        df_with_date = df.copy()
        df_with_date['date'] = df_with_date['timestamp'].dt.date
        df_with_date['is_negative'] = negative_mask
        daily_negative = df_with_date.groupby('date')['is_negative'].sum()
        avg_hours_per_day = daily_negative.mean()
        
        # Find maximum consecutive hours
        max_consecutive = self._find_max_consecutive_hours(negative_mask)
        
        # Monthly breakdown
        df_with_month = df.copy()
        df_with_month['month'] = df_with_month['timestamp'].dt.month
        monthly_breakdown = {}
        
        for month in range(1, 13):
            month_data = df_with_month[df_with_month['month'] == month]
            if not month_data.empty:
                month_negative = (month_data['price'] < 0).sum()
                month_near_zero = (month_data['price'] <= near_zero_threshold).sum()
                month_total = len(month_data)
                
                monthly_breakdown[month] = {
                    'negative_hours': month_negative,
                    'near_zero_hours': month_near_zero,
                    'total_hours': month_total,
                    'negative_percentage': (month_negative / month_total) * 100,
                    'near_zero_percentage': (month_near_zero / month_total) * 100,
                    'avg_hours_per_day': month_negative / (month_total / 24) if month_total > 0 else 0
                }
        
        # Hourly breakdown (time of day patterns)
        df_with_hour = df.copy()
        df_with_hour['hour'] = df_with_hour['timestamp'].dt.hour
        hourly_breakdown = {}
        
        for hour in range(24):
            hour_data = df_with_hour[df_with_hour['hour'] == hour]
            if not hour_data.empty:
                hour_negative = (hour_data['price'] < 0).sum()
                hour_near_zero = (hour_data['price'] <= near_zero_threshold).sum()
                hour_total = len(hour_data)
                
                hourly_breakdown[hour] = {
                    'negative_hours': hour_negative,
                    'near_zero_hours': hour_near_zero,
                    'total_hours': hour_total,
                    'negative_percentage': (hour_negative / hour_total) * 100,
                    'near_zero_percentage': (hour_near_zero / hour_total) * 100
                }
        
        return NegativePricingMetrics(
            total_hours=total_hours,
            negative_hours=negative_hours,
            near_zero_hours=near_zero_hours,
            negative_percentage=negative_percentage,
            near_zero_percentage=near_zero_percentage,
            avg_hours_per_day=avg_hours_per_day,
            max_consecutive_hours=max_consecutive,
            monthly_breakdown=monthly_breakdown,
            hourly_breakdown=hourly_breakdown
        )
    
    def estimate_solar_saturation_potential(self, month: int) -> Dict[str, float]:
        """
        Estimate the theoretical maximum hours of negative pricing based on solar potential.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Dictionary with saturation estimates
        """
        if self.region not in self.solar_potential:
            return {'error': f'No solar data available for region {self.region}'}
        
        # Get solar irradiation for the month (kWh/m²/day)
        monthly_solar = self.solar_potential[self.region][month - 1]
        
        # Get current solar capacity
        current_capacity = self.current_solar_capacity.get(self.region, 0)
        
        # Get grid flexibility factor
        flexibility = self.grid_flexibility.get(self.region, 0.5)
        
        # Estimate peak solar hours per day (when irradiation > 50% of daily max)
        peak_solar_hours = min(max(monthly_solar / 2, 2), 8)  # 2-8 hours range
        
        # Calculate theoretical maximum based on grid saturation
        # Assumption: negative prices occur when solar > 70% of demand during peak hours
        saturation_threshold = 0.7
        
        # Estimate when grid reaches saturation point
        # Higher solar potential = more hours of possible oversupply
        theoretical_max_hours = peak_solar_hours * (monthly_solar / 8) * flexibility
        theoretical_max_hours = min(theoretical_max_hours, 12)  # Cap at 12 hours/day
        
        # Calculate potential with different solar capacity scenarios
        capacity_scenarios = {
            'current': current_capacity,
            '2x_current': current_capacity * 2,
            '5x_current': current_capacity * 5,
            '10x_current': current_capacity * 10
        }
        
        scenario_results = {}
        for scenario, capacity in capacity_scenarios.items():
            # Model: more capacity = more hours of negative pricing, but with diminishing returns
            capacity_factor = capacity / max(current_capacity, 1)
            hours_estimate = theoretical_max_hours * (1 - np.exp(-capacity_factor / 3))
            scenario_results[scenario] = min(hours_estimate, theoretical_max_hours)
        
        return {
            'month': month,
            'solar_irradiation_kwh_m2_day': monthly_solar,
            'theoretical_max_hours_per_day': theoretical_max_hours,
            'grid_flexibility_factor': flexibility,
            'capacity_scenarios': scenario_results,
            'peak_solar_hours_estimate': peak_solar_hours
        }
    
    def calculate_progress_metrics(
        self,
        current_metrics: NegativePricingMetrics,
        month: int
    ) -> Dict[str, float]:
        """
        Calculate progress toward maximum renewable saturation.
        
        Args:
            current_metrics: Current negative pricing metrics
            month: Month number for solar potential calculation
            
        Returns:
            Dictionary with progress metrics
        """
        saturation_data = self.estimate_solar_saturation_potential(month)
        
        if 'error' in saturation_data:
            return saturation_data
        
        current_avg = current_metrics.avg_hours_per_day
        theoretical_max = saturation_data['theoretical_max_hours_per_day']
        
        # Progress calculations
        progress_percentage = (current_avg / theoretical_max) * 100 if theoretical_max > 0 else 0
        remaining_potential = max(0, theoretical_max - current_avg)
        
        # Estimate years to saturation based on current growth trends
        # This would need historical data for accurate projection
        estimated_years_to_saturation = remaining_potential / max(current_avg * 0.1, 0.1)  # Assume 10% annual growth
        
        return {
            'current_hours_per_day': current_avg,
            'theoretical_max_hours_per_day': theoretical_max,
            'progress_percentage': min(progress_percentage, 100),
            'remaining_potential_hours': remaining_potential,
            'estimated_years_to_saturation': min(estimated_years_to_saturation, 50),
            'saturation_level': 'Low' if progress_percentage < 30 else 'Medium' if progress_percentage < 70 else 'High'
        }
    
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Analyze seasonal patterns in negative pricing across all months.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dictionary with seasonal analysis results
        """
        seasonal_analysis = {}
        
        # Analyze each month
        for month in range(1, 13):
            month_data = df[df['timestamp'].dt.month == month]
            
            if not month_data.empty:
                month_metrics = self.analyze_negative_pricing_patterns(month_data)
                saturation_potential = self.estimate_solar_saturation_potential(month)
                progress_metrics = self.calculate_progress_metrics(month_metrics, month)
                
                seasonal_analysis[month] = {
                    'current_metrics': month_metrics,
                    'saturation_potential': saturation_potential,
                    'progress_metrics': progress_metrics
                }
        
        # Calculate yearly summary
        if seasonal_analysis:
            yearly_avg_current = np.mean([data['progress_metrics']['current_hours_per_day'] 
                                        for data in seasonal_analysis.values() 
                                        if 'error' not in data['progress_metrics']])
            
            yearly_avg_potential = np.mean([data['progress_metrics']['theoretical_max_hours_per_day'] 
                                          for data in seasonal_analysis.values() 
                                          if 'error' not in data['progress_metrics']])
            
            seasonal_analysis['yearly_summary'] = {
                'avg_current_hours_per_day': yearly_avg_current,
                'avg_theoretical_max_hours_per_day': yearly_avg_potential,
                'overall_progress_percentage': (yearly_avg_current / yearly_avg_potential) * 100 if yearly_avg_potential > 0 else 0
            }
        
        return seasonal_analysis
    
    def _empty_metrics(self) -> NegativePricingMetrics:
        """Return empty metrics object."""
        return NegativePricingMetrics(
            total_hours=0,
            negative_hours=0,
            near_zero_hours=0,
            negative_percentage=0.0,
            near_zero_percentage=0.0,
            avg_hours_per_day=0.0,
            max_consecutive_hours=0,
            monthly_breakdown={},
            hourly_breakdown={}
        )
    
    def _find_max_consecutive_hours(self, boolean_series: pd.Series) -> int:
        """Find maximum consecutive True values in a boolean series."""
        if boolean_series.empty:
            return 0
        
        # Convert to list and find consecutive runs
        values = boolean_series.values
        max_consecutive = 0
        current_consecutive = 0
        
        for value in values:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive


def analyze_negative_pricing_comprehensive(
    df: pd.DataFrame,
    region: str,
    near_zero_threshold: float = 5.0
) -> Dict[str, any]:
    """
    Convenience function for comprehensive negative pricing analysis.
    
    Args:
        df: DataFrame with electricity price data
        region: Region/country code
        near_zero_threshold: Threshold for near-zero pricing
        
    Returns:
        Dictionary with complete analysis results
    """
    analyzer = NegativePricingAnalyzer(region)
    
    # Overall analysis
    overall_metrics = analyzer.analyze_negative_pricing_patterns(df, near_zero_threshold)
    
    # Seasonal analysis
    seasonal_patterns = analyzer.analyze_seasonal_patterns(df)
    
    return {
        'overall_metrics': overall_metrics,
        'seasonal_patterns': seasonal_patterns,
        'analysis_params': {
            'region': region,
            'near_zero_threshold': near_zero_threshold,
            'data_points': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if not df.empty else None,
                'end': df['timestamp'].max().isoformat() if not df.empty else None
            }
        }
    }