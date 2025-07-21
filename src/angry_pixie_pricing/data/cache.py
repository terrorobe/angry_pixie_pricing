"""Generic caching layer for electricity market data."""

import calendar
from abc import ABC, abstractmethod
from contextlib import suppress
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class CacheableData(Protocol):
    """Protocol for data that can be cached."""

    def to_csv(self, path: Path, **kwargs: Any) -> None:
        """Write data to CSV file."""
        ...

    @classmethod
    def read_csv(cls, path: Path, **kwargs: Any) -> "CacheableData":
        """Read data from CSV file."""
        ...


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    @abstractmethod
    def get_cache_periods(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Get optimal cache periods for the date range."""

    @abstractmethod
    def get_cache_filename(
        self,
        source: str,
        data_type: str,
        region: str,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        """Generate cache filename for the data."""


class MonthlyGranularCacheStrategy(CacheStrategy):
    """Cache strategy that uses monthly granularity for past data."""

    def get_cache_periods(
        self,
        start_date: datetime,
        end_date: datetime,
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
            if current.year < today.year or (current.year == today.year and current.month < today.month):
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
                month_start = current
                next_month = month_start.replace(day=28) + timedelta(days=4)
                month_end = next_month - timedelta(days=next_month.day)
                month_end = month_end.replace(hour=23, minute=59, second=59)

                # Limit to requested end date
                actual_end = min(end_date, month_end)
                periods.append((month_start, actual_end))
                break

        return periods

    def get_cache_filename(
        self,
        source: str,
        data_type: str,
        region: str,
        start_date: datetime,
        end_date: datetime,
    ) -> str:
        """Generate descriptive cache filename with data type."""
        today = date.today()
        start_dt = start_date.date()

        # Normalize source name and region
        source_normalized = source.lower()
        region_normalized = region.upper()

        # Check if this is a whole month
        month_start = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_day = calendar.monthrange(start_date.year, start_date.month)[1]
        month_end = start_date.replace(day=last_day, hour=23, minute=59, second=59)

        if (
            start_date == month_start
            and end_date.date() >= month_end.date()
            and (start_dt.year < today.year or (start_dt.year == today.year and start_dt.month < today.month))
        ):
            # Past whole month
            return f"{source_normalized}_{data_type}_{region_normalized}_{start_date.strftime('%Y-%m')}.csv.gz"
        # Day-based or partial month
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        return f"{source_normalized}_{data_type}_{region_normalized}_{start_str}_to_{end_str}.csv.gz"


class DataCacheManager:
    """Generic cache manager for electricity market data."""

    def __init__(
        self,
        cache_dir: str | Path,
        strategy: CacheStrategy | None = None,
        compression: str = "gzip",
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for caching data
            strategy: Cache strategy to use (defaults to MonthlyGranularCacheStrategy)
            compression: Compression method for CSV files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy or MonthlyGranularCacheStrategy()
        self.compression = compression

    def get_cached_data(
        self,
        source: str,
        data_type: str,
        region: str,
        start_date: datetime,
        end_date: datetime,
        parser_kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame | None:
        """
        Retrieve cached data if available.

        Args:
            source: Data source name
            data_type: Type of data (e.g., 'prices', 'load')
            region: Region code
            start_date: Start date for data
            end_date: End date for data
            parser_kwargs: Additional arguments for CSV parsing

        Returns:
            Cached DataFrame or None if not available
        """
        periods = self.strategy.get_cache_periods(start_date, end_date)
        all_data = []

        for period_start, period_end in periods:
            period_data = self._get_cached_period_data(
                source, data_type, region, period_start, period_end, parser_kwargs,
            )
            if period_data is None:
                return None  # If any period is missing, return None to force full refetch
            all_data.append(period_data)

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            # Filter to exact requested range
            mask = (combined_data["timestamp"] >= start_date) & (combined_data["timestamp"] <= end_date)
            return combined_data[mask].reset_index(drop=True)

        return None

    def cache_data(
        self,
        source: str,
        data_type: str,
        region: str,
        start_date: datetime,
        end_date: datetime,
        data: pd.DataFrame,
        writer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Cache data for a specific period.

        Args:
            source: Data source name
            data_type: Type of data (e.g., 'prices', 'load')
            region: Region code
            start_date: Start date for data
            end_date: End date for data
            data: DataFrame to cache
            writer_kwargs: Additional arguments for CSV writing
        """
        periods = self.strategy.get_cache_periods(start_date, end_date)

        for period_start, period_end in periods:
            # Filter data to this period
            mask = (data["timestamp"] >= period_start) & (data["timestamp"] <= period_end)
            period_data = data[mask]

            if not period_data.empty:
                self._cache_period_data(source, data_type, region, period_start, period_end, period_data, writer_kwargs)

    def _get_cached_period_data(
        self,
        source: str,
        data_type: str,
        region: str,
        start_date: datetime,
        end_date: datetime,
        parser_kwargs: dict[str, Any] | None = None,
    ) -> pd.DataFrame | None:
        """Retrieve cached data for a specific period."""
        cache_filename = self.strategy.get_cache_filename(source, data_type, region, start_date, end_date)
        cache_file = self.cache_dir / cache_filename

        if cache_file.exists():
            try:
                kwargs = {"compression": self.compression, "parse_dates": ["timestamp"]}
                if parser_kwargs:
                    kwargs.update(parser_kwargs)
                return pd.read_csv(cache_file, **kwargs)
            except (pd.errors.ParserError, OSError, ValueError):
                # If cache is corrupted, remove it
                cache_file.unlink(missing_ok=True)

        return None

    def _cache_period_data(
        self,
        source: str,
        data_type: str,
        region: str,
        start_date: datetime,
        end_date: datetime,
        data: pd.DataFrame,
        writer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Cache the data for a specific period."""
        cache_filename = self.strategy.get_cache_filename(source, data_type, region, start_date, end_date)
        cache_file = self.cache_dir / cache_filename

        with suppress(Exception):
            kwargs = {
                "compression": self.compression,
                "index": False,
                "date_format": "%Y-%m-%d %H:%M:%S",
            }
            if writer_kwargs:
                kwargs.update(writer_kwargs)
            data.to_csv(cache_file, **kwargs)

    def clear_cache(
        self,
        source: str | None = None,
        data_type: str | None = None,
        region: str | None = None,
    ) -> None:
        """
        Clear cached data with optional filtering.

        Args:
            source: If specified, only clear cache for this source
            data_type: If specified, only clear cache for this data type
            region: If specified, only clear cache for this region
        """
        if source is None and data_type is None and region is None:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*.csv.gz"):
                cache_file.unlink(missing_ok=True)
        else:
            # Build pattern for specific filters
            pattern_parts = []
            if source:
                pattern_parts.append(source.lower())
            else:
                pattern_parts.append("*")

            if data_type:
                pattern_parts.append(data_type.lower())
            else:
                pattern_parts.append("*")

            if region:
                pattern_parts.append(region.upper())
            else:
                pattern_parts.append("*")

            pattern_parts.append("*.csv.gz")
            pattern = "_".join(pattern_parts)

            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink(missing_ok=True)

    def migrate_legacy_cache_files(self, source: str, data_type: str) -> dict[str, int]:
        """
        Migrate legacy cache files to new naming convention.

        Args:
            source: Source name to migrate
            data_type: Data type to add to filenames

        Returns:
            Dictionary with migration statistics
        """
        source_lower = source.lower()
        legacy_pattern = f"{source_lower}_*.csv.gz"

        migrated = 0
        skipped = 0
        errors = 0

        for cache_file in self.cache_dir.glob(legacy_pattern):
            # Skip files already in new format
            if f"_{data_type}_" in cache_file.name:
                skipped += 1
                continue

            try:
                # Parse legacy filename: source_region_period.csv.gz
                parts = cache_file.stem.replace(".csv", "").split("_")
                if len(parts) >= 3:
                    # Insert data_type after source
                    new_parts = [parts[0], data_type, *parts[1:]]
                    new_filename = "_".join(new_parts) + ".csv.gz"
                    new_path = cache_file.parent / new_filename

                    if not new_path.exists():
                        cache_file.rename(new_path)
                        migrated += 1
                    else:
                        # New file already exists, remove legacy
                        cache_file.unlink()
                        skipped += 1
                else:
                    errors += 1
            except (OSError, ValueError):
                errors += 1

        return {"migrated": migrated, "skipped": skipped, "errors": errors}

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cached data."""
        cache_files = list(self.cache_dir.glob("*.csv.gz"))
        total_size = sum(f.stat().st_size for f in cache_files)

        # Group by source and data type
        by_source_type: dict[str, dict[str, int]] = {}
        for cache_file in cache_files:
            parts = cache_file.stem.replace(".csv", "").split("_")
            if len(parts) >= 2:
                source = parts[0]
                data_type = parts[1] if len(parts) > 2 else "unknown"

                if source not in by_source_type:
                    by_source_type[source] = {}
                by_source_type[source][data_type] = by_source_type[source].get(data_type, 0) + 1

        return {
            "total_files": len(cache_files),
            "total_size_mb": total_size / 1024 / 1024,
            "cache_dir": str(self.cache_dir),
            "by_source_type": by_source_type,
        }
