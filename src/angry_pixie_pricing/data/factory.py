"""Factory for creating data source instances."""

from typing import ClassVar

from .base import PriceDataSource
from .energy_charts import EnergyChartsDataSource


class DataSourceFactory:
    """Factory for creating data source instances."""

    AVAILABLE_SOURCES: ClassVar[dict[str, type[PriceDataSource]]] = {
        "energy-charts": EnergyChartsDataSource,
        "default": EnergyChartsDataSource,  # Default source
    }

    @classmethod
    def create_data_source(
        cls,
        source_name: str = "default",
        cache_dir: str | None = None,
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
        if source_name not in cls.AVAILABLE_SOURCES:
            available = ", ".join(cls.AVAILABLE_SOURCES.keys())
            msg = f"Unknown data source '{source_name}'. Available: {available}"
            raise ValueError(msg)

        source_class = cls.AVAILABLE_SOURCES[source_name]
        return source_class(cache_dir=cache_dir)

    @classmethod
    def list_available_sources(cls) -> list[str]:
        """Get list of available data source names."""
        return [name for name in cls.AVAILABLE_SOURCES if name != "default"]
