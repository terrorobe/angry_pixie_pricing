"""Hourly electricity price analysis for duck curve detection.

This module analyzes electricity prices by hour-of-day to identify patterns
like duck curves, which show characteristic dips during midday solar generation
and peaks during morning/evening demand.
"""

import numpy as np
import pandas as pd

from .day_types import add_day_type_column


class HourlyPriceAnalyzer:
    """Analyzer for hourly electricity price patterns."""

    def __init__(self, region: str):
        """
        Initialize analyzer for a specific region.

        Args:
            region: Region/country code for holiday detection
        """
        self.region = region.upper()

    def analyze_hourly_patterns(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Analyze hourly price patterns split by day type.

        Args:
            df: DataFrame with columns ['timestamp', 'price', 'unit']

        Returns:
            Dictionary with keys 'workday', 'non_workday', 'all' containing
            DataFrames with hourly statistics
        """
        # Add day type classifications
        df_with_types = add_day_type_column(df, self.region)

        # Split into workday and non-workday data
        workday_df = df_with_types[df_with_types["is_workday"]].copy()
        non_workday_df = df_with_types[df_with_types["is_non_workday"]].copy()

        results = {}

        # Analyze each dataset
        if not workday_df.empty:
            results["workday"] = self._calculate_hourly_stats(workday_df)

        if not non_workday_df.empty:
            results["non_workday"] = self._calculate_hourly_stats(non_workday_df)

        if not df_with_types.empty:
            results["all"] = self._calculate_hourly_stats(df_with_types)

        return results

    def _calculate_hourly_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate hourly statistics for a dataset."""
        # Extract hour from timestamp
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour

        # Calculate statistics by hour
        hourly_stats = (
            df.groupby("hour")["price"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index()
        )

        # Calculate confidence intervals (95%)
        hourly_stats["ci_lower"] = hourly_stats["mean"] - 1.96 * (
            hourly_stats["std"] / np.sqrt(hourly_stats["count"])
        )
        hourly_stats["ci_upper"] = hourly_stats["mean"] + 1.96 * (
            hourly_stats["std"] / np.sqrt(hourly_stats["count"])
        )

        # Add percentage of negative prices by hour
        negative_by_hour = (
            df[df["price"] < 0].groupby("hour").size().reindex(range(24), fill_value=0)
        )
        total_by_hour = (
            df.groupby("hour").size().reindex(range(24), fill_value=1)
        )  # Avoid division by zero
        hourly_stats["negative_price_pct"] = (negative_by_hour / total_by_hour * 100).values

        return hourly_stats.fillna(0)

    def detect_duck_curve_features(self, hourly_stats: pd.DataFrame) -> dict[str, float]:
        """
        Detect characteristic duck curve features.

        Args:
            hourly_stats: DataFrame from _calculate_hourly_stats

        Returns:
            Dictionary with duck curve metrics
        """
        if hourly_stats.empty:
            return {}

        prices = hourly_stats["mean"].values
        hours = hourly_stats["hour"].values

        # Find morning peak (6-10 AM)
        morning_mask = (hours >= 6) & (hours <= 10)
        morning_peak_hour = (
            hours[morning_mask][np.argmax(prices[morning_mask])] if morning_mask.any() else None
        )
        morning_peak_price = np.max(prices[morning_mask]) if morning_mask.any() else None

        # Find midday minimum (10 AM - 3 PM)
        midday_mask = (hours >= 10) & (hours <= 15)
        midday_min_hour = (
            hours[midday_mask][np.argmin(prices[midday_mask])] if midday_mask.any() else None
        )
        midday_min_price = np.min(prices[midday_mask]) if midday_mask.any() else None

        # Find evening peak (5-9 PM)
        evening_mask = (hours >= 17) & (hours <= 21)
        evening_peak_hour = (
            hours[evening_mask][np.argmax(prices[evening_mask])] if evening_mask.any() else None
        )
        evening_peak_price = np.max(prices[evening_mask]) if evening_mask.any() else None

        # Calculate duck curve depth (morning peak to midday minimum)
        duck_depth = None
        if morning_peak_price is not None and midday_min_price is not None:
            duck_depth = morning_peak_price - midday_min_price

        # Calculate evening ramp (midday minimum to evening peak)
        evening_ramp = None
        if midday_min_price is not None and evening_peak_price is not None:
            evening_ramp = evening_peak_price - midday_min_price

        return {
            "morning_peak_hour": morning_peak_hour,
            "morning_peak_price": morning_peak_price,
            "midday_min_hour": midday_min_hour,
            "midday_min_price": midday_min_price,
            "evening_peak_hour": evening_peak_hour,
            "evening_peak_price": evening_peak_price,
            "duck_depth": duck_depth,
            "evening_ramp": evening_ramp,
            "price_range": np.max(prices) - np.min(prices),
            "avg_negative_price_pct": hourly_stats["negative_price_pct"].mean(),
        }

    def compare_workday_vs_nonworkday(
        self, analysis_results: dict[str, pd.DataFrame]
    ) -> dict[str, any]:
        """
        Compare duck curve characteristics between workdays and non-workdays.

        Args:
            analysis_results: Results from analyze_hourly_patterns

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}

        if "workday" in analysis_results and "non_workday" in analysis_results:
            workday_features = self.detect_duck_curve_features(analysis_results["workday"])
            nonworkday_features = self.detect_duck_curve_features(analysis_results["non_workday"])

            comparison["workday_features"] = workday_features
            comparison["non_workday_features"] = nonworkday_features

            # Calculate differences
            if workday_features and nonworkday_features:
                comparison["differences"] = {
                    "duck_depth_diff": (workday_features.get("duck_depth", 0) or 0)
                    - (nonworkday_features.get("duck_depth", 0) or 0),
                    "evening_ramp_diff": (workday_features.get("evening_ramp", 0) or 0)
                    - (nonworkday_features.get("evening_ramp", 0) or 0),
                    "price_range_diff": (workday_features.get("price_range", 0) or 0)
                    - (nonworkday_features.get("price_range", 0) or 0),
                }

        return comparison


def analyze_hourly_patterns(df: pd.DataFrame, region: str) -> dict[str, pd.DataFrame]:
    """
    Convenience function to analyze hourly patterns for a region.

    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region/country code

    Returns:
        Dictionary with hourly analysis results
    """
    analyzer = HourlyPriceAnalyzer(region)
    return analyzer.analyze_hourly_patterns(df)


def detect_duck_curve_strength(df: pd.DataFrame, region: str) -> float:
    """
    Calculate a simple duck curve strength metric (0-1).

    Args:
        df: DataFrame with electricity price data
        region: Region/country code

    Returns:
        Duck curve strength score (higher = more pronounced duck curve)
    """
    analyzer = HourlyPriceAnalyzer(region)
    analysis = analyzer.analyze_hourly_patterns(df)

    if "workday" not in analysis:
        return 0.0

    features = analyzer.detect_duck_curve_features(analysis["workday"])

    # Simple scoring based on duck depth and evening ramp
    duck_depth = features.get("duck_depth", 0) or 0
    evening_ramp = features.get("evening_ramp", 0) or 0
    price_range = features.get("price_range", 1) or 1  # Avoid division by zero

    # Normalize by price range to make score comparable across regions/time periods
    strength = (duck_depth + evening_ramp) / (2 * price_range)

    return min(max(strength, 0.0), 1.0)  # Clamp to 0-1 range
