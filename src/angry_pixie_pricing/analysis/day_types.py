"""Day type classification for electricity price analysis.

This module provides utilities to classify dates as workdays, weekends, or holidays
for European electricity market analysis. The key insight is that weekends and
holidays show similar electricity consumption patterns distinct from workdays.
"""

from datetime import date, datetime
from typing import Literal

import holidays
import pandas as pd

# Type definitions
DayType = Literal["workday", "weekend", "holiday"]


class DayTypeClassifier:
    """Classifier for determining day types based on holidays and weekdays."""

    def __init__(self, country_code: str):
        """
        Initialize classifier for a specific country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code (e.g., 'DE', 'AT', 'FR')
        """
        self.country_code = country_code.upper()
        self._holiday_cache: dict[int, set[date]] = {}

    def _get_holidays_for_year(self, year: int) -> set[date]:
        """Get holidays for a specific year, with caching."""
        if year not in self._holiday_cache:
            country_holidays = holidays.country_holidays(self.country_code, years=year)
            self._holiday_cache[year] = set(country_holidays.keys())
        return self._holiday_cache[year]

    def classify_day_type(self, timestamp: datetime) -> DayType:
        """
        Classify a timestamp as workday, weekend, or holiday.

        Args:
            timestamp: The datetime to classify

        Returns:
            'holiday' if the date is a public holiday
            'weekend' if the date is Saturday or Sunday
            'workday' if the date is Monday-Friday and not a holiday
        """
        dt = timestamp.date()

        # Check if it's a public holiday
        year_holidays = self._get_holidays_for_year(dt.year)
        if dt in year_holidays:
            return "holiday"

        # Check if it's a weekend (Saturday=5, Sunday=6)
        if timestamp.weekday() >= 5:
            return "weekend"

        return "workday"

    def is_workday(self, timestamp: datetime) -> bool:
        """Check if timestamp falls on a workday."""
        return self.classify_day_type(timestamp) == "workday"

    def is_non_workday(self, timestamp: datetime) -> bool:
        """Check if timestamp falls on weekend or holiday."""
        return self.classify_day_type(timestamp) in ("weekend", "holiday")


def classify_day_type(timestamp: datetime, country_code: str) -> DayType:
    """
    Standalone function to classify a single timestamp.

    Args:
        timestamp: The datetime to classify
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'DE', 'AT', 'FR')

    Returns:
        'holiday', 'weekend', or 'workday'
    """
    classifier = DayTypeClassifier(country_code)
    return classifier.classify_day_type(timestamp)


def add_day_type_column(
    df: pd.DataFrame, country_code: str, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Add day type classification columns to a DataFrame.

    Args:
        df: DataFrame with timestamp column
        country_code: ISO 3166-1 alpha-2 country code
        timestamp_col: Name of the timestamp column

    Returns:
        DataFrame with added columns:
        - 'day_type': 'workday', 'weekend', or 'holiday'
        - 'is_workday': Boolean indicating if it's a workday
        - 'is_non_workday': Boolean indicating if it's weekend/holiday
    """
    classifier = DayTypeClassifier(country_code)

    # Add day type classification
    df = df.copy()
    df["day_type"] = df[timestamp_col].apply(classifier.classify_day_type)
    df["is_workday"] = df["day_type"] == "workday"
    df["is_non_workday"] = df["day_type"].isin(["weekend", "holiday"])

    return df


def get_workday_vs_nonworkday_split(
    df: pd.DataFrame, country_code: str, timestamp_col: str = "timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into workday and non-workday (weekend/holiday) portions.

    Args:
        df: DataFrame with timestamp column
        country_code: ISO 3166-1 alpha-2 country code
        timestamp_col: Name of the timestamp column

    Returns:
        Tuple of (workday_df, non_workday_df)
    """
    df_with_types = add_day_type_column(df, country_code, timestamp_col)

    workday_df = df_with_types[df_with_types["is_workday"]].copy()
    non_workday_df = df_with_types[df_with_types["is_non_workday"]].copy()

    return workday_df, non_workday_df


def analyze_day_type_distribution(
    df: pd.DataFrame, country_code: str, timestamp_col: str = "timestamp"
) -> dict[str, int]:
    """
    Analyze the distribution of day types in a dataset.

    Args:
        df: DataFrame with timestamp column
        country_code: ISO 3166-1 alpha-2 country code
        timestamp_col: Name of the timestamp column

    Returns:
        Dictionary with counts for each day type
    """
    df_with_types = add_day_type_column(df, country_code, timestamp_col)

    return {
        "workday": int(df_with_types["is_workday"].sum()),
        "weekend": int((df_with_types["day_type"] == "weekend").sum()),
        "holiday": int((df_with_types["day_type"] == "holiday").sum()),
        "total": len(df_with_types),
    }
