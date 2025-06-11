"""Abstract base class for electricity price data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import hashlib
import pickle
import os


class PriceDataSource(ABC):
    """Abstract base class for electricity price data sources with caching."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data source with optional cache directory.

        Args:
            cache_dir: Directory for caching data. Defaults to ./data/cache/
        """
        if cache_dir is None:
            cache_dir = Path.cwd() / "data" / "cache"
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

        # Fetch fresh data
        data = self._fetch_spot_prices(region, start_date, end_date)

        # Cache the data
        if use_cache:
            self._cache_data(region, start_date, end_date, data)

        return data

    def _get_cache_key(
        self, region: str, start_date: datetime, end_date: datetime
    ) -> str:
        """Generate a cache key for the given parameters."""
        key_string = f"{self.__class__.__name__}_{region}_{start_date.isoformat()}_{end_date.isoformat()}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cached_data(
        self, region: str, start_date: datetime, end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available."""
        cache_key = self._get_cache_key(region, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                # If cache is corrupted, remove it
                cache_file.unlink(missing_ok=True)

        return None

    def _cache_data(
        self, region: str, start_date: datetime, end_date: datetime, data: pd.DataFrame
    ):
        """Cache the data for future use."""
        cache_key = self._get_cache_key(region, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            # If caching fails, continue without error
            pass

    def clear_cache(self, region: Optional[str] = None):
        """
        Clear cached data.

        Args:
            region: If specified, only clear cache for this region. Otherwise clear all.
        """
        if region is None:
            # Clear all cache files for this data source
            prefix = f"{self.__class__.__name__}_"
            for cache_file in self.cache_dir.glob(f"{prefix}*.pkl"):
                cache_file.unlink(missing_ok=True)
        else:
            # Clear cache files for specific region
            prefix = f"{self.__class__.__name__}_{region}_"
            for cache_file in self.cache_dir.glob(f"{prefix}*.pkl"):
                cache_file.unlink(missing_ok=True)

    @abstractmethod
    def _fetch_spot_prices(
        self, region: str, start_date: datetime, end_date: datetime
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
        pass

    @abstractmethod
    def get_supported_regions(self) -> List[str]:
        """
        Get list of supported region/country codes.

        Returns:
            List of supported region codes
        """
        pass

    @abstractmethod
    def get_data_source_info(self) -> Dict[str, str]:
        """
        Get information about the data source.

        Returns:
            Dictionary with source name, license, attribution, etc.
        """
        pass
