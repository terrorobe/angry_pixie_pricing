"""Simple wrapper for accessing electricity price data."""

from datetime import datetime

import pandas as pd

from .factory import DataSourceFactory


def get_electricity_prices(
    region: str,
    start_date: datetime,
    end_date: datetime,
    use_cache: bool = True,
    data_source: str = "energy-charts",
) -> pd.DataFrame:
    """Get electricity spot prices for a region and date range.

    Args:
        region: Region code (e.g., 'DE', 'FR', 'AT')
        start_date: Start date
        end_date: End date
        use_cache: Whether to use cached data
        data_source: Data source to use

    Returns:
        DataFrame with DatetimeIndex and 'Price' column (EUR/MWh)
    """
    # Create data source
    source = DataSourceFactory.create_data_source(data_source)

    # Get spot prices
    df = source.get_spot_prices(region, start_date, end_date, use_cache=use_cache)

    # Transform to expected format: DatetimeIndex with 'Price' column
    if not df.empty:
        df = df.set_index("timestamp")
        df = df.rename(columns={"price": "Price"})

    return df
