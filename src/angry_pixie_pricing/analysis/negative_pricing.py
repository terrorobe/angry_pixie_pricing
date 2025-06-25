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
            current_values = [data['progress_metrics']['current_hours_per_day'] 
                            for data in seasonal_analysis.values() 
                            if 'error' not in data['progress_metrics']]
            potential_values = [data['progress_metrics']['theoretical_max_hours_per_day'] 
                              for data in seasonal_analysis.values() 
                              if 'error' not in data['progress_metrics']]
            
            yearly_avg_current = np.mean(current_values) if current_values else 0
            yearly_avg_potential = np.mean(potential_values) if potential_values else 0
            
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


def calculate_daily_hours_timeseries(
    df: pd.DataFrame,
    near_zero_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Calculate daily hours with negative/near-zero prices for timechart visualization.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        near_zero_threshold: Price threshold for "near-zero" classification (EUR/MWh)
        
    Returns:
        DataFrame with columns ['date', 'negative_hours', 'near_zero_hours']
    """
    if df.empty:
        return pd.DataFrame(columns=['date', 'negative_hours', 'near_zero_hours'])
    
    # Add date column and calculate daily aggregations
    df_daily = df.copy()
    df_daily['date'] = df_daily['timestamp'].dt.date
    df_daily['is_negative'] = df_daily['price'] < 0
    df_daily['is_near_zero'] = df_daily['price'] <= near_zero_threshold
    
    # Group by date and count hours
    daily_counts = df_daily.groupby('date').agg({
        'is_negative': 'sum',
        'is_near_zero': 'sum'
    }).reset_index()
    
    daily_counts.columns = ['date', 'negative_hours', 'near_zero_hours']
    
    # Convert date back to datetime for plotting
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
    
    return daily_counts.sort_values('date')


def calculate_weekly_hours_timeseries(
    df: pd.DataFrame,
    near_zero_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Calculate weekly average daily hours with negative/near-zero prices for timechart visualization.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        near_zero_threshold: Price threshold for "near-zero" classification (EUR/MWh)
        
    Returns:
        DataFrame with columns ['week_start', 'negative_hours', 'near_zero_hours']
        where hours represent average daily hours within each week
    """
    if df.empty:
        return pd.DataFrame(columns=['week_start', 'negative_hours', 'near_zero_hours'])
    
    # Add weekly grouping columns
    df_weekly = df.copy()
    df_weekly['week_start'] = df_weekly['timestamp'].dt.to_period('W').dt.start_time
    df_weekly['is_negative'] = df_weekly['price'] < 0
    df_weekly['is_near_zero'] = df_weekly['price'] <= near_zero_threshold
    
    # Group by week and calculate total hours and days
    weekly_agg = df_weekly.groupby('week_start').agg({
        'is_negative': 'sum',
        'is_near_zero': 'sum',
        'timestamp': 'count'  # Total hours in the week
    }).reset_index()
    
    # Calculate average daily hours within each week
    weekly_agg['days_in_week'] = weekly_agg['timestamp'] / 24  # Convert hours to days
    weekly_agg['negative_hours'] = weekly_agg['is_negative'] / weekly_agg['days_in_week']
    weekly_agg['near_zero_hours'] = weekly_agg['is_near_zero'] / weekly_agg['days_in_week']
    
    # Return only the columns we need
    result = weekly_agg[['week_start', 'negative_hours', 'near_zero_hours']].copy()
    
    return result.sort_values('week_start')


def calculate_monthly_hours_timeseries(
    df: pd.DataFrame,
    near_zero_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Calculate monthly average daily hours with negative/near-zero prices for timechart visualization.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        near_zero_threshold: Price threshold for "near-zero" classification (EUR/MWh)
        
    Returns:
        DataFrame with columns ['month_start', 'negative_hours', 'near_zero_hours']
        where hours represent average daily hours within each month
    """
    if df.empty:
        return pd.DataFrame(columns=['month_start', 'negative_hours', 'near_zero_hours'])
    
    # Add monthly grouping columns
    df_monthly = df.copy()
    df_monthly['month_start'] = df_monthly['timestamp'].dt.to_period('M').dt.start_time
    df_monthly['is_negative'] = df_monthly['price'] < 0
    df_monthly['is_near_zero'] = df_monthly['price'] <= near_zero_threshold
    
    # Group by month and calculate total hours and days
    monthly_agg = df_monthly.groupby('month_start').agg({
        'is_negative': 'sum',
        'is_near_zero': 'sum',
        'timestamp': 'count'  # Total hours in the month
    }).reset_index()
    
    # Calculate average daily hours within each month
    monthly_agg['days_in_month'] = monthly_agg['timestamp'] / 24  # Convert hours to days
    monthly_agg['negative_hours'] = monthly_agg['is_negative'] / monthly_agg['days_in_month']
    monthly_agg['near_zero_hours'] = monthly_agg['is_near_zero'] / monthly_agg['days_in_month']
    
    # Return only the columns we need
    result = monthly_agg[['month_start', 'negative_hours', 'near_zero_hours']].copy()
    
    return result.sort_values('month_start')


def calculate_solar_quarter_hours_timeseries(
    df: pd.DataFrame,
    near_zero_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Calculate solar quarter average daily hours with negative/near-zero prices for timechart visualization.
    
    Solar quarters based on European solar irradiation patterns:
    - Peak Sun (May-Jul): Peak irradiation months
    - Rising Sun (Feb-Apr): Solar ramp-up period  
    - Fading Sun (Aug-Oct): Solar decline but still significant
    - Low Sun (Nov-Jan): Minimal irradiation
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        near_zero_threshold: Price threshold for "near-zero" classification (EUR/MWh)
        
    Returns:
        DataFrame with columns ['quarter_start', 'quarter_name', 'negative_hours', 'near_zero_hours']
        where hours represent average daily hours within each solar quarter
    """
    if df.empty:
        return pd.DataFrame(columns=['quarter_start', 'quarter_name', 'negative_hours', 'near_zero_hours'])
    
    # Define solar quarter mapping
    def get_solar_quarter(month):
        """Map calendar month to solar quarter."""
        if month in [5, 6, 7]:  # May-Jul
            return ('Peak Sun', 2)  # Q2
        elif month in [2, 3, 4]:  # Feb-Apr  
            return ('Rising Sun', 1)  # Q1
        elif month in [8, 9, 10]:  # Aug-Oct
            return ('Fading Sun', 3)  # Q3
        else:  # Nov, Dec, Jan
            return ('Low Sun', 4)  # Q4
    
    # Add solar quarter columns
    df_solar = df.copy()
    df_solar['month'] = df_solar['timestamp'].dt.month
    df_solar['year'] = df_solar['timestamp'].dt.year
    df_solar[['quarter_name', 'quarter_num']] = df_solar['month'].apply(
        lambda m: pd.Series(get_solar_quarter(m))
    )
    
    # Create quarter start dates (using standard quarters for consistency)
    # Q1: Feb-Apr -> Jan 1, Q2: May-Jul -> Apr 1, Q3: Aug-Oct -> Jul 1, Q4: Nov-Jan -> Oct 1
    quarter_start_months = {1: 1, 2: 4, 3: 7, 4: 10}  # Approximate quarter starts
    df_solar['quarter_start'] = df_solar.apply(
        lambda row: pd.Timestamp(row['year'], quarter_start_months[row['quarter_num']], 1),
        axis=1
    )
    
    # Handle year transitions for Q4 (Nov-Jan spans years)
    # For Nov-Dec, use current year. For Jan, use previous year's Q4
    mask_jan = df_solar['month'] == 1
    df_solar.loc[mask_jan, 'quarter_start'] = df_solar.loc[mask_jan].apply(
        lambda row: pd.Timestamp(row['year'] - 1, 10, 1), axis=1
    )
    
    # Add boolean columns for negative and near-zero pricing
    df_solar['is_negative'] = df_solar['price'] < 0
    df_solar['is_near_zero'] = df_solar['price'] <= near_zero_threshold
    
    # Group by quarter and calculate aggregations
    quarter_agg = df_solar.groupby(['quarter_start', 'quarter_name']).agg({
        'is_negative': 'sum',
        'is_near_zero': 'sum',
        'timestamp': 'count'  # Total hours in the quarter
    }).reset_index()
    
    # Calculate average daily hours within each quarter
    quarter_agg['days_in_quarter'] = quarter_agg['timestamp'] / 24  # Convert hours to days
    quarter_agg['negative_hours'] = quarter_agg['is_negative'] / quarter_agg['days_in_quarter']
    quarter_agg['near_zero_hours'] = quarter_agg['is_near_zero'] / quarter_agg['days_in_quarter']
    
    # Return only the columns we need
    result = quarter_agg[['quarter_start', 'quarter_name', 'negative_hours', 'near_zero_hours']].copy()
    
    return result.sort_values('quarter_start')


def calculate_aggregated_hours_timeseries(
    df: pd.DataFrame,
    aggregation_level: str = "daily",
    near_zero_threshold: float = 5.0
) -> pd.DataFrame:
    """
    Calculate aggregated hours with negative/near-zero prices for timechart visualization.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        aggregation_level: Aggregation level - "daily", "weekly", "monthly", or "solar-quarters"
        near_zero_threshold: Price threshold for "near-zero" classification (EUR/MWh)
        
    Returns:
        DataFrame with appropriate time column and ['negative_hours', 'near_zero_hours']
    """
    if aggregation_level == "daily":
        result = calculate_daily_hours_timeseries(df, near_zero_threshold)
        # Rename date column to time_period for consistency
        result = result.rename(columns={'date': 'time_period'})
    elif aggregation_level == "weekly":
        result = calculate_weekly_hours_timeseries(df, near_zero_threshold)
        result = result.rename(columns={'week_start': 'time_period'})
    elif aggregation_level == "monthly":
        result = calculate_monthly_hours_timeseries(df, near_zero_threshold)
        result = result.rename(columns={'month_start': 'time_period'})
    elif aggregation_level == "solar-quarters":
        result = calculate_solar_quarter_hours_timeseries(df, near_zero_threshold)
        result = result.rename(columns={'quarter_start': 'time_period'})
        # Keep quarter_name column for chart labeling
    else:
        raise ValueError(f"Invalid aggregation_level: {aggregation_level}. Must be 'daily', 'weekly', 'monthly', or 'solar-quarters'")
    
    return result


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