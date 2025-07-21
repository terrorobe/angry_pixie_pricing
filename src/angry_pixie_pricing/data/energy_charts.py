"""Energy-Charts.info data source implementation."""

from datetime import datetime
from typing import ClassVar

import pandas as pd
import requests

from .base import PriceDataSource


class EnergyChartsDataSource(PriceDataSource):
    """Data source for energy-charts.info API."""

    BASE_URL = "https://api.energy-charts.info"

    # Map common country codes to energy-charts bidding zones
    REGION_MAPPING: ClassVar[dict[str, str]] = {
        "DE": "DE-LU",  # Germany-Luxembourg
        "AT": "AT",  # Austria
        "BE": "BE",  # Belgium
        "CH": "CH",  # Switzerland
        "FR": "FR",  # France (if available)
        "NL": "NL",  # Netherlands (if available)
        "DK": "DK1",  # Denmark (if available)
    }

    def _fetch_spot_prices(
        self, region: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch hourly spot prices from energy-charts.info API.

        Args:
            region: Region/country code (e.g., 'DE', 'AT', 'FR')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval

        Returns:
            DataFrame with columns: ['timestamp', 'price', 'unit']
        """
        # Map region code to bidding zone
        bzn = self.REGION_MAPPING.get(region.upper(), region)

        # Format dates for API
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Build API URL
        url = f"{self.BASE_URL}/price"
        params = {"bzn": bzn, "start": start_str, "end": end_str}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check if data is available
            if not data.get("unix_seconds") or not data.get("price"):
                raise ValueError(
                    f"No price data available for {region} ({bzn}) from {start_str} to {end_str}"
                )

            # Convert to DataFrame
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(data["unix_seconds"], unit="s"),
                    "price": data["price"],
                    "unit": data.get("unit", "EUR/MWh"),
                }
            )

            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch data from energy-charts.info: {e}") from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid response from energy-charts.info API: {e}") from e

    def get_supported_regions(self) -> list[str]:
        """
        Get list of supported region/country codes.

        Returns:
            List of supported region codes
        """
        return list(self.REGION_MAPPING.keys())

    def get_data_source_info(self) -> dict[str, str]:
        """
        Get information about the data source.

        Returns:
            Dictionary with source name, license, attribution, etc.
        """
        return {
            "name": "Energy-Charts.info",
            "url": "https://www.energy-charts.info",
            "api_url": self.BASE_URL,
            "license": "CC BY 4.0",
            "attribution": "Bundesnetzagentur | SMARD.de",
            "description": "European electricity spot market prices",
            "data_unit": "EUR/MWh",
            "frequency": "Hourly",
        }
