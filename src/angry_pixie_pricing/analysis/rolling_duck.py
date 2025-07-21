"""Rolling duck curve analysis for long-term trend detection.

This module provides rolling window analysis of duck curve patterns
to detect trends in renewable energy impact on electricity pricing.
"""

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .hourly import HourlyPriceAnalyzer, detect_duck_curve_strength


class RollingDuckAnalyzer:
    """Analyzer for rolling duck curve patterns over extended time periods."""

    def __init__(self, region: str):
        """
        Initialize rolling duck analyzer for a specific region.

        Args:
            region: Region/country code for analysis
        """
        self.region = region.upper()
        self.hourly_analyzer = HourlyPriceAnalyzer(region)

    def calculate_rolling_duck_factor(
        self, df: pd.DataFrame, window_days: int = 30, step_days: int = 7,
    ) -> pd.DataFrame:
        """
        Calculate rolling duck factor over time with configurable windows.

        Args:
            df: DataFrame with columns ['timestamp', 'price', 'unit']
            window_days: Size of rolling window in days
            step_days: Step size between calculations in days

        Returns:
            DataFrame with columns ['date', 'duck_factor', 'window_days', 'data_points']
        """
        if df.empty:
            return pd.DataFrame(columns=["date", "duck_factor", "window_days", "data_points"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Get date range
        start_date = df["timestamp"].min().date()
        end_date = df["timestamp"].max().date()

        results = []
        current_date = start_date + timedelta(days=window_days)

        while current_date <= end_date:
            # Define window
            window_start = current_date - timedelta(days=window_days)
            window_end = current_date + timedelta(days=1)  # Include end date

            # Filter data for window
            window_mask = (df["timestamp"].dt.date >= window_start) & (
                df["timestamp"].dt.date < window_end
            )
            window_df = df[window_mask].copy()

            if len(window_df) >= 24:  # Need at least one day of data
                duck_factor = detect_duck_curve_strength(window_df, self.region)

                results.append(
                    {
                        "date": current_date,
                        "duck_factor": duck_factor,
                        "window_days": window_days,
                        "data_points": len(window_df),
                    },
                )

            current_date += timedelta(days=step_days)

        return pd.DataFrame(results)

    def multi_window_analysis(
        self, df: pd.DataFrame, windows: list[int] | None = None, step_days: int = 7,
    ) -> dict[str, pd.DataFrame]:
        """
        Calculate duck factors for multiple window sizes.

        Args:
            df: DataFrame with price data
            windows: List of window sizes in days
            step_days: Step size between calculations

        Returns:
            Dictionary mapping window size to duck factor DataFrame
        """
        if windows is None:
            windows = [7, 30, 90]
        results = {}

        for window_days in windows:
            window_key = f"{window_days}d"
            results[window_key] = self.calculate_rolling_duck_factor(df, window_days, step_days)

        return results

    def detect_seasonal_patterns(self, duck_factors_df: pd.DataFrame) -> dict[str, Any]:
        """
        Detect seasonal patterns in duck factor time series.

        Args:
            duck_factors_df: DataFrame from calculate_rolling_duck_factor

        Returns:
            Dictionary with seasonal analysis results
        """
        if duck_factors_df.empty:
            return {}

        # Add date components
        df = duck_factors_df.copy()
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter

        # Monthly averages
        monthly_avg = df.groupby("month")["duck_factor"].agg(["mean", "std", "count"]).reset_index()
        monthly_avg["season"] = monthly_avg["month"].map(self._get_season)

        # Quarterly averages
        quarterly_avg = (
            df.groupby("quarter")["duck_factor"].agg(["mean", "std", "count"]).reset_index()
        )

        # Seasonal averages
        seasonal_avg = (
            df.groupby(df["month"].map(self._get_season))["duck_factor"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        seasonal_avg.columns = ["season", "mean", "std", "count"]

        # Find peak seasonal difference
        seasonal_range = (
            seasonal_avg["mean"].max() - seasonal_avg["mean"].min() if not seasonal_avg.empty else 0
        )

        return {
            "monthly_patterns": monthly_avg,
            "quarterly_patterns": quarterly_avg,
            "seasonal_patterns": seasonal_avg,
            "seasonal_range": seasonal_range,
            "peak_season": seasonal_avg.loc[seasonal_avg["mean"].idxmax(), "season"]
            if not seasonal_avg.empty
            else None,
            "low_season": seasonal_avg.loc[seasonal_avg["mean"].idxmin(), "season"]
            if not seasonal_avg.empty
            else None,
        }

    def detect_trends(
        self, duck_factors_df: pd.DataFrame, min_data_points: int = 20,
    ) -> dict[str, Any]:
        """
        Detect long-term trends in duck factor evolution.

        Args:
            duck_factors_df: DataFrame from calculate_rolling_duck_factor
            min_data_points: Minimum points needed for trend analysis

        Returns:
            Dictionary with trend analysis results
        """
        if len(duck_factors_df) < min_data_points:
            return {
                "error": (
                    f"Insufficient data points for trend analysis "
                    f"(need {min_data_points}, got {len(duck_factors_df)})"
                ),
            }

        df = duck_factors_df.copy()
        df["date_numeric"] = (
            pd.to_datetime(df["date"]).astype(np.int64) / 10**9
        )  # Convert to unix timestamp

        # Linear trend
        try:
            trend_coef = np.polyfit(df["date_numeric"], df["duck_factor"], 1)
            trend_slope = trend_coef[0] * (365.25 * 24 * 3600)  # Per year

            # R-squared for trend quality
            y_pred = np.polyval(trend_coef, df["date_numeric"])
            ss_res = np.sum((df["duck_factor"] - y_pred) ** 2)
            ss_tot = np.sum((df["duck_factor"] - df["duck_factor"].mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        except (ValueError, np.linalg.LinAlgError):
            trend_slope = 0
            r_squared = 0

        # Volatility analysis
        volatility = df["duck_factor"].std()
        rolling_volatility = (
            df["duck_factor"].rolling(window=min(10, len(df) // 4), center=True).std()
        )
        volatility_trend = 0

        if len(rolling_volatility.dropna()) > 5:
            try:
                vol_dates = df.loc[rolling_volatility.notna(), "date_numeric"]
                vol_coef = np.polyfit(vol_dates, rolling_volatility.dropna(), 1)
                volatility_trend = vol_coef[0] * (365.25 * 24 * 3600)  # Per year
            except (ValueError, np.linalg.LinAlgError):
                pass

        # Inflection points (significant changes in trend)
        inflection_points = self._find_inflection_points(df)

        return {
            "trend_slope_per_year": trend_slope,
            "trend_r_squared": r_squared,
            "overall_volatility": volatility,
            "volatility_trend_per_year": volatility_trend,
            "inflection_points": inflection_points,
            "data_span_years": (df["date"].max() - df["date"].min()).days / 365.25,
            "trend_classification": self._classify_trend(trend_slope, r_squared),
        }

    def year_over_year_comparison(self, duck_factors_df: pd.DataFrame) -> dict[str, Any]:
        """
        Compare duck factors year-over-year to identify evolution patterns.

        Args:
            duck_factors_df: DataFrame from calculate_rolling_duck_factor

        Returns:
            Dictionary with year-over-year comparison results
        """
        if duck_factors_df.empty:
            return {}

        df = duck_factors_df.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month

        # Annual averages
        annual_avg = df.groupby("year")["duck_factor"].agg(["mean", "std", "count"]).reset_index()

        # Year-over-year changes
        annual_avg["yoy_change"] = annual_avg["mean"].pct_change()
        annual_avg["yoy_change_abs"] = annual_avg["mean"].diff()

        # Monthly year-over-year for same months
        monthly_yoy = []
        for month in range(1, 13):
            month_data = df[df["month"] == month].copy()
            if len(month_data) >= 2:
                month_annual = month_data.groupby("year")["duck_factor"].mean()
                month_yoy_change = month_annual.pct_change().dropna()

                if not month_yoy_change.empty:
                    monthly_yoy.append(
                        {
                            "month": month,
                            "avg_yoy_change": month_yoy_change.mean(),
                            "max_yoy_change": month_yoy_change.max(),
                            "min_yoy_change": month_yoy_change.min(),
                            "years_compared": len(month_yoy_change),
                        },
                    )

        monthly_yoy_df = pd.DataFrame(monthly_yoy)

        return {
            "annual_averages": annual_avg,
            "monthly_yoy_patterns": monthly_yoy_df,
            "avg_annual_change": annual_avg["yoy_change"].mean()
            if "yoy_change" in annual_avg.columns
            else 0,
            "peak_growth_year": annual_avg.loc[annual_avg["yoy_change"].idxmax(), "year"]
            if not annual_avg.empty and "yoy_change" in annual_avg.columns
            else None,
            "years_analyzed": len(annual_avg),
        }

    def _get_season(self, month: int) -> str:
        """Map month number to season name."""
        if month in [12, 1, 2]:
            return "Winter"
        if month in [3, 4, 5]:
            return "Spring"
        if month in [6, 7, 8]:
            return "Summer"
        return "Fall"

    def _find_inflection_points(
        self, df: pd.DataFrame, sensitivity: float = 0.1,
    ) -> list[dict[str, Any]]:
        """Find significant changes in duck factor trends."""
        if len(df) < 10:
            return []

        # Calculate rolling trend (slope) over small windows
        window_size = max(5, len(df) // 10)
        trends = []

        for i in range(window_size, len(df) - window_size):
            subset = df.iloc[i - window_size : i + window_size]
            try:
                x = pd.to_datetime(subset["date"]).astype(np.int64) / 10**9
                coef = np.polyfit(x, subset["duck_factor"], 1)
                trends.append(coef[0])
            except (ValueError, np.linalg.LinAlgError):
                trends.append(0)

        # Find points where trend changes significantly
        inflection_points = []
        for i in range(1, len(trends)):
            change = abs(trends[i] - trends[i - 1])
            if change > sensitivity:
                point_idx = i + window_size
                if point_idx < len(df):
                    inflection_points.append(
                        {
                            "date": df.iloc[point_idx]["date"],
                            "duck_factor": df.iloc[point_idx]["duck_factor"],
                            "trend_change": change,
                            "trend_before": trends[i - 1],
                            "trend_after": trends[i],
                        },
                    )

        return inflection_points

    def _classify_trend(self, slope: float, r_squared: float) -> str:
        """Classify trend based on slope and R-squared."""
        if r_squared < 0.3:
            return "No Clear Trend"
        if abs(slope) < 0.01:
            return "Stable"
        if slope > 0.05:
            return "Strong Upward"
        if slope > 0.01:
            return "Moderate Upward"
        if slope < -0.05:
            return "Strong Downward"
        if slope < -0.01:
            return "Moderate Downward"
        return "Weak Trend"


def analyze_rolling_duck_patterns(
    df: pd.DataFrame, region: str, window_days: int = 30, step_days: int = 7,
) -> dict[str, Any]:
    """
    Convenience function for comprehensive rolling duck analysis.

    Args:
        df: DataFrame with electricity price data
        region: Region/country code
        window_days: Rolling window size in days
        step_days: Step size between calculations

    Returns:
        Dictionary with complete rolling analysis results
    """
    analyzer = RollingDuckAnalyzer(region)

    # Calculate rolling duck factors
    duck_factors = analyzer.calculate_rolling_duck_factor(df, window_days, step_days)

    if duck_factors.empty:
        return {"error": "No data available for analysis"}

    # Perform all analyses
    seasonal_patterns = analyzer.detect_seasonal_patterns(duck_factors)
    trends = analyzer.detect_trends(duck_factors)
    yoy_comparison = analyzer.year_over_year_comparison(duck_factors)

    return {
        "duck_factors": duck_factors,
        "seasonal_patterns": seasonal_patterns,
        "trends": trends,
        "year_over_year": yoy_comparison,
        "analysis_params": {
            "window_days": window_days,
            "step_days": step_days,
            "region": region,
            "data_points": len(df),
            "date_range": {
                "start": df["timestamp"].min().isoformat() if not df.empty else None,
                "end": df["timestamp"].max().isoformat() if not df.empty else None,
            },
        },
    }
