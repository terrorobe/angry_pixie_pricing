"""Load profile from actual smart meter data."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .base import LoadProfile


class SmartMeterProfile(LoadProfile):
    """Load profile from smart meter CSV data."""

    def __init__(
        self,
        csv_path: Path,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        timestamp_col: str = "timestamp",
        power_col: str | None = None,
        energy_col: str | None = None,
        delimiter: str = ",",
        decimal: str = ".",
        encoding: str = "utf-8",
        date_format: str | None = None,
    ):
        """Initialize smart meter profile from CSV.

        Args:
            csv_path: Path to CSV file with smart meter data
            start_date: Optional start date filter
            end_date: Optional end date filter
            timestamp_col: Name of timestamp column
            power_col: Name of power column (kW). If None, calculated from energy
            energy_col: Name of energy column (kWh). If None, calculated from power
            delimiter: CSV delimiter
            decimal: Decimal separator
            encoding: File encoding
            date_format: Date format string for parsing timestamps
        """
        self.csv_path = Path(csv_path)
        self.timestamp_col = timestamp_col
        self.power_col = power_col
        self.energy_col = energy_col
        self.delimiter = delimiter
        self.decimal = decimal
        self.encoding = encoding
        self.date_format = date_format

        # Load data to determine actual date range
        raw_data = self._load_raw_data()

        # Use provided dates or data range
        super().__init__(start_date or raw_data.index.min(), end_date or raw_data.index.max())

        self._raw_data = raw_data

    def _load_raw_data(self) -> pd.DataFrame:
        """Load and parse raw CSV data."""
        # Read CSV
        df = pd.read_csv(
            self.csv_path,
            delimiter=self.delimiter,
            decimal=self.decimal,
            encoding=self.encoding,
        )

        # Parse timestamps
        if self.date_format:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], format=self.date_format)
        else:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])

        # Set timestamp as index
        df = df.set_index(self.timestamp_col)
        return df.sort_index()

    def get_profile(self) -> pd.DataFrame:
        """Get the load profile from smart meter data."""
        # Filter by date range
        mask = (self._raw_data.index >= self.start_date) & (self._raw_data.index <= self.end_date)
        data = self._raw_data[mask].copy()

        # Ensure we have both power and energy columns
        if self.power_col and self.energy_col:
            # Both provided
            data["power_kw"] = data[self.power_col]
            data["energy_kwh"] = data[self.energy_col]
        elif self.power_col:
            # Only power provided, calculate energy (assuming 15-min intervals)
            data["power_kw"] = data[self.power_col]
            data["energy_kwh"] = data["power_kw"] * 0.25  # 15 minutes = 0.25 hours
        elif self.energy_col:
            # Only energy provided, calculate power
            data["energy_kwh"] = data[self.energy_col]
            data["power_kw"] = data["energy_kwh"] * 4  # 15 minutes = 0.25 hours
        else:
            msg = "Either power_col or energy_col must be specified"
            raise ValueError(msg)

        # Resample to ensure consistent 15-minute intervals
        result = pd.DataFrame(
            index=pd.date_range(start=self.start_date, end=self.end_date, freq="15min"),
        )

        # Forward fill missing values (common in smart meter data)
        result["power_kw"] = data["power_kw"].reindex(result.index).ffill().bfill()
        result["energy_kwh"] = data["energy_kwh"].reindex(result.index).ffill().bfill()

        # Handle any remaining NaN values
        return result.fillna(0)

    @classmethod
    def from_standard_formats(
        cls,
        csv_path: Path,
        format_type: str = "auto",
        **kwargs: Any,
    ) -> "SmartMeterProfile":
        """Create profile from common smart meter formats.

        Args:
            csv_path: Path to CSV file
            format_type: Format type ('auto', 'discovergy', 'tibber', 'generic')
            **kwargs: Additional arguments passed to constructor

        Returns:
            SmartMeterProfile instance
        """
        format_configs = {
            "discovergy": {
                "timestamp_col": "timestamp",
                "energy_col": "energy_kwh",
                "delimiter": ";",
                "decimal": ",",
                "date_format": "%Y-%m-%d %H:%M:%S",
            },
            "tibber": {
                "timestamp_col": "from",
                "energy_col": "consumption",
                "delimiter": ",",
                "decimal": ".",
                "date_format": "%Y-%m-%dT%H:%M:%S%z",
            },
            "generic": {
                "timestamp_col": "timestamp",
                "power_col": "power",
                "delimiter": ",",
                "decimal": ".",
            },
        }

        if format_type == "auto":
            # Try to detect format from file
            format_type = cls._detect_format(csv_path)

        config = format_configs.get(format_type, format_configs["generic"])
        config.update(kwargs)  # Allow overrides

        return cls(csv_path, **config)  # type: ignore[arg-type]

    @staticmethod
    def _detect_format(csv_path: Path) -> str:
        """Try to detect smart meter format from CSV headers."""
        # Read first few lines
        with open(csv_path) as f:
            header = f.readline().strip()

        # Check for known patterns
        if "discovergy" in header.lower() or ";" in header:
            return "discovergy"
        if "tibber" in header.lower() or "from,to,consumption" in header:
            return "tibber"
        return "generic"

    def validate_data(self) -> dict[str, Any]:
        """Validate smart meter data quality."""
        df = self.data

        # Check for gaps
        expected_intervals = pd.date_range(start=self.start_date, end=self.end_date, freq="15min")
        missing_intervals = expected_intervals.difference(df.index)

        # Check for negative values
        negative_power = (df["power_kw"] < 0).sum()
        negative_energy = (df["energy_kwh"] < 0).sum()

        # Check for unrealistic values
        unrealistic_power = (df["power_kw"] > 1000).sum()  # > 1 MW for residential

        return {
            "total_intervals": len(df),
            "expected_intervals": len(expected_intervals),
            "missing_intervals": len(missing_intervals),
            "missing_percentage": len(missing_intervals) / len(expected_intervals) * 100,
            "negative_power_count": negative_power,
            "negative_energy_count": negative_energy,
            "unrealistic_power_count": unrealistic_power,
            "data_quality_score": min(
                100,
                100 * (1 - len(missing_intervals) / len(expected_intervals)),
            ),
        }
