"""Factory for creating data source instances."""

from typing import Optional
from .base import PriceDataSource
from .energy_charts import EnergyChartsDataSource


class DataSourceFactory:
    """Factory for creating data source instances."""

    AVAILABLE_SOURCES = {
        "energy-charts": EnergyChartsDataSource,
        "default": EnergyChartsDataSource,  # Default source
    }

    @classmethod
    def create_data_source(
        self, source_name: str = "default", cache_dir: Optional[str] = None
    ) -> PriceDataSource:
        """
        Create a data source instance.

        Args:
            source_name: Name of the data source ('energy-charts', 'default')
            cache_dir: Optional cache directory path

        Returns:
            PriceDataSource instance

        Raises:
            ValueError: If source_name is not supported
        """
        if source_name not in self.AVAILABLE_SOURCES:
            available = ", ".join(self.AVAILABLE_SOURCES.keys())
            raise ValueError(
                f"Unknown data source '{source_name}'. Available: {available}"
            )

        source_class = self.AVAILABLE_SOURCES[source_name]
        return source_class(cache_dir=cache_dir)

    @classmethod
    def list_available_sources(cls) -> list[str]:
        """Get list of available data source names."""
        return [name for name in cls.AVAILABLE_SOURCES.keys() if name != "default"]
