"""Abstract base class for electricity price data sources."""

import calendar
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


class PriceDataSource(ABC):
    """Abstract base class for electricity price data sources with caching."""

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the data source with optional cache directory.

        Args:
            cache_dir: Directory for caching data. Defaults to ./data/cache/
        """
        if cache_dir is None:
            self.cache_dir = Path.cwd() / "data" / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_spot_prices(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get hourly spot prices for a region and date range with caching.

        Args:
            region: Region/country code (e.g., 'DE', 'AT', 'FR')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with columns: ['timestamp', 'price', 'unit']
        """
        if use_cache:
            cached_data = self._get_cached_data(region, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # Determine optimal fetch/cache periods
        fetch_periods = self._get_cache_periods(start_date, end_date)

        all_data = []
        for period_start, period_end in fetch_periods:
            # Check cache for this period
            period_data = None
            if use_cache:
                period_data = self._get_cached_period_data(region, period_start, period_end)

            if period_data is None:
                # Fetch fresh data for this period
                period_data = self._fetch_spot_prices(region, period_start, period_end)

                # Cache the data for this period
                if use_cache:
                    self._cache_period_data(region, period_start, period_end, period_data)

            all_data.append(period_data)

        # Combine all periods and filter to requested range
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Filter to exact requested range
            mask = (combined_data["timestamp"] >= start_date) & (
                combined_data["timestamp"] <= end_date
            )
            return combined_data[mask].reset_index(drop=True)
        return pd.DataFrame(columns=["timestamp", "price", "unit"])

    def _get_cache_periods(
        self, start_date: datetime, end_date: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """
        Split date range into optimal cache periods.
        Past months are cached whole, current month is cached by day.
        """
        periods = []
        current = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        today = date.today()

        while current <= end_date:
            # Determine if this is a past month or current month
            if current.year < today.year or (
                current.year == today.year and current.month < today.month
            ):
                # Past month - cache entire month
                last_day = calendar.monthrange(current.year, current.month)[1]
                period_end = current.replace(day=last_day, hour=23, minute=59, second=59)
                periods.append((current, period_end))
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            else:
                # Current month - cache by day
                # Find the range of days we need in this month
                month_start = current
                next_month = month_start.replace(day=28) + timedelta(days=4)
                month_end = next_month - timedelta(days=next_month.day)
                month_end = month_end.replace(hour=23, minute=59, second=59)

                # Limit to requested end date
                actual_end = min(end_date, month_end)

                # For current month, we'll still fetch the whole month but cache by day
                # This simplifies the logic while keeping cache granular
                periods.append((month_start, actual_end))
                break

        return periods

    def _get_cache_filename(self, region: str, start_date: datetime, end_date: datetime) -> str:
        """Generate readable cache filename with data source attribution."""
        today = date.today()
        start_dt = start_date.date()

        # Get a clean data source name for filename and normalize region case
        source_name = self.__class__.__name__.replace("DataSource", "").lower()
        region_normalized = region.upper()

        # Check if this is a whole month
        month_start = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day = calendar.monthrange(start_date.year, start_date.month)[1]
        month_end = start_date.replace(day=last_day, hour=23, minute=59, second=59)

        if (
            start_date == month_start
            and end_date.date() >= month_end.date()
            and (
                start_dt.year < today.year
                or (start_dt.year == today.year and start_dt.month < today.month)
            )
        ):
            # Past whole month
            return f"{source_name}_{region_normalized}_{start_date.strftime('%Y-%m')}.csv.gz"
        # Day-based or partial month
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        return f"{source_name}_{region_normalized}_{start_str}_to_{end_str}.csv.gz"

    def _get_cached_data(
        self, region: str, start_date: datetime, end_date: datetime,
    ) -> pd.DataFrame | None:
        """Retrieve cached data if available (legacy method for backwards compatibility)."""
        # This method is kept for backwards compatibility but will use the new period-based logic
        periods = self._get_cache_periods(start_date, end_date)
        all_data = []

        for period_start, period_end in periods:
            period_data = self._get_cached_period_data(region, period_start, period_end)
            if period_data is None:
                return None  # If any period is missing, return None to force full refetch
            all_data.append(period_data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Filter to exact requested range
            mask = (combined_data["timestamp"] >= start_date) & (
                combined_data["timestamp"] <= end_date
            )
            return combined_data[mask].reset_index(drop=True)

        return None

    def _get_cached_period_data(
        self, region: str, start_date: datetime, end_date: datetime,
    ) -> pd.DataFrame | None:
        """Retrieve cached data for a specific period."""
        cache_filename = self._get_cache_filename(region, start_date, end_date)
        cache_file = self.cache_dir / cache_filename

        if cache_file.exists():
            try:
                # Read CSV with gzip compression
                return pd.read_csv(cache_file, compression="gzip", parse_dates=["timestamp"])
            except (pd.errors.ParserError, OSError, ValueError):
                # If cache is corrupted, remove it
                cache_file.unlink(missing_ok=True)

        return None

    def _cache_period_data(
        self, region: str, start_date: datetime, end_date: datetime, data: pd.DataFrame,
    ) -> None:
        """Cache the data for a specific period."""
        cache_filename = self._get_cache_filename(region, start_date, end_date)
        cache_file = self.cache_dir / cache_filename

        from contextlib import suppress

        with suppress(Exception):
            # Write CSV with gzip compression
            data.to_csv(
                cache_file, compression="gzip", index=False, date_format="%Y-%m-%d %H:%M:%S",
            )

    def clear_cache(self, region: str | None = None) -> None:
        """
        Clear cached data.

        Args:
            region: If specified, only clear cache for this region. Otherwise clear all.
        """
        if region is None:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.csv.gz"):
                cache_file.unlink(missing_ok=True)
        else:
            # Clear cache files for specific region
            source_name = self.__class__.__name__.replace("DataSource", "").lower()
            region_normalized = region.upper()
            pattern = f"{source_name}_{region_normalized}_*.csv.gz"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink(missing_ok=True)

    @abstractmethod
    def _fetch_spot_prices(
        self, region: str, start_date: datetime, end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch hourly spot prices from the data source (no caching).

        Args:
            region: Region/country code (e.g., 'DE', 'AT', 'FR')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval

        Returns:
            DataFrame with columns: ['timestamp', 'price', 'unit']
        """

    @abstractmethod
    def get_supported_regions(self) -> list[str]:
        """
        Get list of supported region/country codes.

        Returns:
            List of supported region codes
        """

    @abstractmethod
    def get_data_source_info(self) -> dict[str, str]:
        """
        Get information about the data source.

        Returns:
            Dictionary with source name, license, attribution, etc.
        """
