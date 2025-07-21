"""Abstract base class for electricity price data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cache import DataCacheManager


class PriceDataSource(ABC):
    """Abstract base class for electricity price data sources with caching."""

    def __init__(self, cache_dir: str | Path | None = None):
        """
        Initialize the data source with optional cache directory.

        Args:
            cache_dir: Directory for caching data. Defaults to ./data/cache/
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / "data" / "cache"
        self.cache_manager = DataCacheManager(cache_dir)
        self.source_name = self.__class__.__name__.replace("DataSource", "").lower()

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
            cached_data = self.cache_manager.get_cached_data(self.source_name, "prices", region, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # Determine optimal fetch/cache periods
        fetch_periods = self.cache_manager.strategy.get_cache_periods(start_date, end_date)

        all_data = []
        for period_start, period_end in fetch_periods:
            # Check cache for this period
            period_data = None
            if use_cache:
                period_data = self.cache_manager._get_cached_period_data(
                    self.source_name, "prices", region, period_start, period_end,
                )

            if period_data is None:
                # Fetch fresh data for this period
                period_data = self._fetch_spot_prices(region, period_start, period_end)

                # Cache the data for this period
                if use_cache and not period_data.empty:
                    self.cache_manager._cache_period_data(
                        self.source_name, "prices", region, period_start, period_end, period_data,
                    )

            all_data.append(period_data)

        # Combine all periods and filter to requested range
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Filter to exact requested range
            mask = (combined_data["timestamp"] >= start_date) & (combined_data["timestamp"] <= end_date)
            return combined_data[mask].reset_index(drop=True)
        return pd.DataFrame(columns=["timestamp", "price", "unit"])

    def get_load_data(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get hourly load data for a region and date range with caching.

        Args:
            region: Region/country code (e.g., 'DE', 'AT', 'FR')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with load data columns
        """
        if use_cache:
            cached_data = self.cache_manager.get_cached_data(self.source_name, "load", region, start_date, end_date)
            if cached_data is not None:
                return cached_data

        # Determine optimal fetch/cache periods
        fetch_periods = self.cache_manager.strategy.get_cache_periods(start_date, end_date)

        all_data = []
        for period_start, period_end in fetch_periods:
            # Check cache for this period
            period_data = None
            if use_cache:
                period_data = self.cache_manager._get_cached_period_data(
                    self.source_name, "load", region, period_start, period_end,
                )

            if period_data is None:
                # Fetch fresh data for this period
                if hasattr(self, "fetch_load_data"):
                    period_data = self.fetch_load_data(region, period_start, period_end)
                else:
                    # Fallback to empty DataFrame if method not implemented
                    period_data = pd.DataFrame()

                # Cache the data for this period
                if use_cache and not period_data.empty:
                    self.cache_manager._cache_period_data(
                        self.source_name, "load", region, period_start, period_end, period_data,
                    )

            all_data.append(period_data)

        # Combine all periods and filter to requested range
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            if not combined_data.empty and "timestamp" in combined_data.columns:
                # Filter to exact requested range
                mask = (combined_data["timestamp"] >= start_date) & (combined_data["timestamp"] <= end_date)
                return combined_data[mask].reset_index(drop=True)
            return combined_data
        return pd.DataFrame()

    def clear_cache(self, region: str | None = None, data_type: str | None = None) -> None:
        """
        Clear cached data.

        Args:
            region: If specified, only clear cache for this region
            data_type: If specified, only clear cache for this data type ('prices', 'load')
        """
        self.cache_manager.clear_cache(self.source_name, data_type, region)

    @abstractmethod
    def _fetch_spot_prices(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
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
