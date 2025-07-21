"""Load pattern analysis for detecting behind-the-meter PV impacts."""

from datetime import datetime

import numpy as np
import pandas as pd

from ..data.energy_charts import EnergyChartsDataSource


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
    ) -> pd.DataFrame:
        """
        Fetch load data across multiple years.

        Args:
            region: Region/country code (e.g., 'DE', 'AT')
            start_year: Starting year for analysis
            end_year: Ending year for analysis (inclusive)

        Returns:
            DataFrame with load data across all requested years
        """
        all_data = []

        for year in range(start_year, end_year + 1):
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)

            try:
                year_data = self.data_source.fetch_load_data(
                    region=region,
                    start_date=start_date,
                    end_date=end_date,
                )
                year_data["year"] = year
                all_data.append(year_data)
                print(f"✓ Fetched {len(year_data)} hours of load data for {region} {year}")
            except (ValueError, ConnectionError) as e:
                print(f"⚠ Failed to fetch {region} {year} data: {e}")
                continue

        if not all_data:
            raise ValueError(f"No load data available for {region} from {start_year}-{end_year}")

        return pd.concat(all_data, ignore_index=True)

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
        peaks = (
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

        return peaks

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
        peaks_pivot = df.pivot(index="hour", columns="year", values="peak_load")

        # Calculate changes relative to reference year
        if reference_year not in peaks_pivot.columns:
            raise ValueError(f"Reference year {reference_year} not found in data")

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
