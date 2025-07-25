"""Load pattern analysis for detecting behind-the-meter PV impacts."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from angry_pixie_pricing.data.energy_charts import EnergyChartsDataSource


class LoadPatternAnalyzer:
    """Analyze electricity load patterns to detect behind-the-meter PV impacts."""

    def __init__(self) -> None:
        """Initialize the load pattern analyzer."""
        self.data_source = EnergyChartsDataSource()

    def fetch_multi_year_load_data(
        self,
        region: str,
        start_year: int,
        end_year: int,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Fetch load data across multiple years with coverage tracking.

        Args:
            region: Region/country code (e.g., 'DE', 'AT')
            start_year: Starting year for analysis
            end_year: Ending year for analysis (inclusive)

        Returns:
            Tuple of (DataFrame with load data, coverage_info dict)
        """
        all_data: list[pd.DataFrame] = []
        coverage_info: dict[str, Any] = {
            "requested_years": list(range(start_year, end_year + 1)),
            "available_years": [],
            "missing_years": [],
            "start_year": start_year,
            "end_year": end_year,
            "actual_start_date": None,
            "actual_end_date": None,
        }

        for year in range(start_year, end_year + 1):
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

            try:
                year_data = self.data_source.get_load_data(
                    region=region,
                    start_date=start_date,
                    end_date=end_date,
                )
                year_data["year"] = year
                all_data.append(year_data)
                coverage_info["available_years"].append(year)
                print(f"✓ Fetched {len(year_data)} hours of load data for {region} {year}")
            except (ValueError, ConnectionError) as e:
                coverage_info["missing_years"].append(year)
                print(f"⚠ Failed to fetch {region} {year} data: {e}")
                continue

        if not all_data:
            msg = f"No load data available for {region} from {start_year}-{end_year}"
            raise ValueError(msg)

        # Determine actual date coverage from the data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Convert timestamp column to datetime if it's not already
        if "timestamp" in combined_data.columns:
            combined_data["timestamp"] = pd.to_datetime(combined_data["timestamp"])
            coverage_info["actual_start_date"] = combined_data["timestamp"].min()
            coverage_info["actual_end_date"] = combined_data["timestamp"].max()
        else:
            # Fallback to year-based tracking if no timestamp column
            coverage_info["actual_start_date"] = datetime(min(coverage_info["available_years"]), 1, 1)
            coverage_info["actual_end_date"] = datetime(max(coverage_info["available_years"]), 12, 31)

        return combined_data, coverage_info

    def calculate_hourly_peaks(
        self,
        df: pd.DataFrame,
        group_by: str = "year",
        percentile: float = 95.0,
    ) -> pd.DataFrame:
        """
        Calculate peak loads for each hour of the day across time periods.

        Args:
            df: DataFrame with load data including 'timestamp', 'load_mw', and grouping column
            group_by: Column to group by ('year', 'month', 'season', etc.)
            percentile: Percentile to use for peak calculation (default 95th percentile)

        Returns:
            DataFrame with hourly peaks by time period
        """
        # Add hour column
        df = df.copy()
        df["hour"] = df["timestamp"].dt.hour

        # Group by time period and hour, calculate peak loads
        return (
            df.groupby([group_by, "hour"])["load_mw"]
            .agg(
                [
                    ("peak_load", lambda x: np.percentile(x.dropna(), percentile)),
                    ("max_load", "max"),
                    ("avg_load", "mean"),
                    ("count", "count"),
                ],
            )
            .reset_index()
        )

    def calculate_daily_peaks(
        self,
        df: pd.DataFrame,
        group_by: str = "year",
    ) -> pd.DataFrame:
        """
        Calculate daily peak loads and their timing.

        Args:
            df: DataFrame with load data
            group_by: Column to group by for analysis

        Returns:
            DataFrame with daily peak statistics by time period
        """
        df = df.copy()
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour

        # Find daily peaks
        daily_peaks = (
            df.groupby(["date", group_by])
            .apply(lambda x: x.loc[x["load_mw"].idxmax()] if len(x) > 0 else None)
            .reset_index(drop=True)
        )

        # Calculate statistics by time period
        peak_stats = (
            daily_peaks.groupby(group_by)
            .agg(
                {
                    "load_mw": ["mean", "std", lambda x: np.percentile(x, 95)],
                    "hour": ["mean", "std", lambda x: np.percentile(x, [25, 50, 75])],
                },
            )
            .reset_index()
        )

        # Flatten column names
        peak_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in peak_stats.columns]

        return peak_stats

    def analyze_peak_migration(
        self,
        df: pd.DataFrame,
        reference_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Analyze how peak timing has changed over years.

        Args:
            df: DataFrame with hourly peaks by year
            reference_year: Year to use as baseline (defaults to earliest year)

        Returns:
            DataFrame showing peak migration patterns
        """
        if reference_year is None:
            reference_year = df["year"].min()

        # Pivot to get peak loads by hour and year
        peaks_pivot = df.pivot_table(index="hour", columns="year", values="peak_load", aggfunc="mean")

        # Calculate changes relative to reference year
        if reference_year not in peaks_pivot.columns:
            msg = f"Reference year {reference_year} not found in data"
            raise ValueError(msg)

        reference_peaks = peaks_pivot[reference_year]

        migration_data = []
        for year in peaks_pivot.columns:
            if year == reference_year:
                continue

            year_peaks = peaks_pivot[year]

            # Calculate peak load changes
            load_changes = ((year_peaks - reference_peaks) / reference_peaks * 100).fillna(0)

            # Find hours with significant changes
            increasing_hours = load_changes[load_changes > 5].index.tolist()
            decreasing_hours = load_changes[load_changes < -5].index.tolist()

            migration_data.append(
                {
                    "year": year,
                    "reference_year": reference_year,
                    "peak_load_change_pct": load_changes.mean(),
                    "max_increase_pct": load_changes.max(),
                    "max_decrease_pct": load_changes.min(),
                    "increasing_hours": increasing_hours,
                    "decreasing_hours": decreasing_hours,
                    "morning_peak_change": load_changes[6:10].mean(),  # 6-9 AM
                    "evening_peak_change": load_changes[17:21].mean(),  # 5-8 PM
                    "midday_change": load_changes[11:15].mean(),  # 11 AM - 2 PM
                },
            )

        return pd.DataFrame(migration_data)

    def detect_duck_curve_formation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze duck curve formation by measuring midday demand suppression.

        Args:
            df: DataFrame with hourly load data by year

        Returns:
            DataFrame with duck curve metrics by year
        """
        duck_metrics = []

        for year in df["year"].unique():
            year_data = df[df["year"] == year].copy()
            year_data["hour"] = year_data["timestamp"].dt.hour

            # Calculate average load by hour
            hourly_avg = year_data.groupby("hour")["load_mw"].mean()

            # Duck curve metrics
            morning_peak = hourly_avg[6:10].max()  # 6-9 AM
            evening_peak = hourly_avg[17:21].max()  # 5-8 PM
            midday_min = hourly_avg[11:15].min()  # 11 AM - 2 PM
            daily_avg = hourly_avg.mean()

            # Duck curve intensity: how deep is the midday dip relative to peaks
            duck_intensity = ((morning_peak + evening_peak) / 2 - midday_min) / daily_avg * 100

            # Peak-to-trough ratio
            peak_trough_ratio = ((morning_peak + evening_peak) / 2) / midday_min

            # Midday suppression: how much below average is midday demand
            midday_suppression = (daily_avg - midday_min) / daily_avg * 100

            duck_metrics.append(
                {
                    "year": year,
                    "morning_peak_mw": morning_peak,
                    "evening_peak_mw": evening_peak,
                    "midday_min_mw": midday_min,
                    "daily_avg_mw": daily_avg,
                    "duck_intensity_pct": duck_intensity,
                    "peak_trough_ratio": peak_trough_ratio,
                    "midday_suppression_pct": midday_suppression,
                },
            )

        return pd.DataFrame(duck_metrics)
