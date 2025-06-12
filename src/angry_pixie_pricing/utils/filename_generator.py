"""Smart filename generation for chart outputs."""

import os
from datetime import datetime
from typing import Optional


def generate_chart_filename(
    chart_type: str,
    region: str,
    start_date: str,
    end_date: str,
    window_days: Optional[int] = None,
    suffix: Optional[str] = None,
    base_dir: str = "images"
) -> str:
    """
    Generate descriptive filename for chart outputs.
    
    Args:
        chart_type: Type of chart (duck-factor, hourly, seasonal, etc.)
        region: Region code (DE, FR, etc.)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        window_days: Rolling window size in days (for rolling analyses)
        suffix: Additional suffix for the filename
        base_dir: Base directory for saving files
        
    Returns:
        Full path to the output file
    """
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Clean dates for filename
    start_clean = start_date.replace("-", "")
    end_clean = end_date.replace("-", "")
    
    # Build filename components
    parts = [chart_type, region.lower(), start_clean, end_clean]
    
    # Add window size for rolling analyses
    if window_days:
        parts.append(f"{window_days}d")
    
    # Add suffix if provided
    if suffix:
        parts.append(suffix)
    
    # Join with underscores and add extension
    filename = "_".join(parts) + ".png"
    
    return os.path.join(base_dir, filename)


def generate_duck_factor_filename(
    region: str,
    start_date: str,
    end_date: str,
    window_days: int,
    chart_subtype: str = "timeseries",
    base_dir: str = "images"
) -> str:
    """
    Generate filename specifically for duck factor charts.
    
    Args:
        region: Region code
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) 
        window_days: Rolling window size
        chart_subtype: Subtype (timeseries, seasonal, multiwindow)
        base_dir: Base directory
        
    Returns:
        Full path for duck factor chart
    """
    return generate_chart_filename(
        chart_type="duck-factor",
        region=region,
        start_date=start_date,
        end_date=end_date,
        window_days=window_days,
        suffix=chart_subtype,
        base_dir=base_dir
    )


def generate_price_chart_filename(
    region: str,
    start_date: str,
    end_date: str,
    chart_type: str = "line",
    base_dir: str = "images"
) -> str:
    """
    Generate filename for price charts.
    
    Args:
        region: Region code
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        chart_type: Type of price chart (line, hourly, workday)
        base_dir: Base directory
        
    Returns:
        Full path for price chart
    """
    chart_name_map = {
        "line": "prices",
        "hourly": "duck-curve", 
        "hourly-workday": "workday-duck"
    }
    
    chart_name = chart_name_map.get(chart_type, chart_type)
    
    return generate_chart_filename(
        chart_type=chart_name,
        region=region,
        start_date=start_date,
        end_date=end_date,
        base_dir=base_dir
    )


def get_multi_window_filenames(
    region: str,
    start_date: str,
    end_date: str,
    windows: list,
    base_dir: str = "images"
) -> dict:
    """
    Generate filenames for multi-window analysis.
    
    Args:
        region: Region code
        start_date: Start date
        end_date: End date
        windows: List of window sizes
        base_dir: Base directory
        
    Returns:
        Dictionary mapping window sizes to filenames
    """
    filenames = {}
    
    for window_days in windows:
        filename = generate_duck_factor_filename(
            region=region,
            start_date=start_date,
            end_date=end_date,
            window_days=window_days,
            chart_subtype=f"{window_days}d-window",
            base_dir=base_dir
        )
        filenames[f"{window_days}d"] = filename
    
    return filenames


def generate_timestamped_filename(
    chart_type: str,
    region: str,
    base_dir: str = "images"
) -> str:
    """
    Generate filename with current timestamp.
    
    Args:
        chart_type: Type of chart
        region: Region code
        base_dir: Base directory
        
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs(base_dir, exist_ok=True)
    filename = f"{chart_type}_{region.lower()}_{timestamp}.png"
    
    return os.path.join(base_dir, filename)