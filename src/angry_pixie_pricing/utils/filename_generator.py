"""Smart filename generation for chart outputs."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _find_project_root() -> str:
    """
    Find the project root directory by looking for setup.py or pyproject.toml.

    Returns:
        Path to project root directory
    """
    current = Path.cwd()

    # Look for project markers up the directory tree
    for parent in [current, *list(current.parents)]:
        if any((parent / marker).exists() for marker in ["setup.py", "pyproject.toml", "CLAUDE.md"]):
            return str(parent)

    # If no project root found, use current directory
    return str(current)


def generate_chart_filename(
    chart_type: str,
    region: str,
    start_date: str,
    end_date: str,
    window_days: int | None = None,
    suffix: str | None = None,
    base_dir: str = "images",
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
        base_dir: Base directory for saving files (relative to project root)

    Returns:
        Full path to the output file
    """
    # Find project root and create absolute path to images directory
    project_root = _find_project_root()
    full_base_dir = os.path.join(project_root, base_dir)

    # Ensure base directory exists
    os.makedirs(full_base_dir, exist_ok=True)

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

    return os.path.join(full_base_dir, filename)


def generate_duck_factor_filename(
    region: str,
    start_date: str,
    end_date: str,
    window_days: int,
    chart_subtype: str = "timeseries",
    base_dir: str = "images",
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
        base_dir=base_dir,
    )


def generate_price_chart_filename(
    region: str,
    start_date: str,
    end_date: str,
    chart_type: str = "line",
    base_dir: str = "images",
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
    chart_name_map = {"line": "prices", "hourly": "duck-curve", "hourly-workday": "workday-duck"}

    chart_name = chart_name_map.get(chart_type, chart_type)

    return generate_chart_filename(
        chart_type=chart_name,
        region=region,
        start_date=start_date,
        end_date=end_date,
        base_dir=base_dir,
    )


def get_multi_window_filenames(
    region: str,
    start_date: str,
    end_date: str,
    windows: list[str],
    base_dir: str = "images",
) -> dict[str, str]:
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

    for window_str in windows:
        # Convert window string like "7d" to integer
        window_days = int(window_str.rstrip("d"))
        filename = generate_duck_factor_filename(
            region=region,
            start_date=start_date,
            end_date=end_date,
            window_days=window_days,
            chart_subtype=f"{window_str}-window",
            base_dir=base_dir,
        )
        filenames[f"{window_str}"] = filename

    return filenames


def generate_load_peaks_filename(
    region: str,
    coverage_info: dict[str, Any],
    chart_subtype: str = "hourly-evolution",
    filter_param: str | None = None,
    base_dir: str = "images",
) -> str:
    """
    Generate filename for load peaks charts using ACTUAL data coverage.

    Args:
        region: Region code
        coverage_info: Coverage info dict with actual_start_date and actual_end_date
        chart_subtype: Subtype (hourly-evolution, duck-curve, peak-migration)
        filter_param: Optional filter parameter (e.g., "p99", "p95", "p90")
        base_dir: Base directory

    Returns:
        Full path for load peaks chart
    """
    start_date = coverage_info["actual_start_date"].strftime("%Y-%m-%d")
    end_date = coverage_info["actual_end_date"].strftime("%Y-%m-%d")

    # Build suffix with optional filter parameter
    suffix_parts = [chart_subtype]
    if filter_param:
        suffix_parts.append(filter_param)
    suffix = "-".join(suffix_parts)

    return generate_chart_filename(
        chart_type="load-peaks",
        region=region,
        start_date=start_date,
        end_date=end_date,
        suffix=suffix,
        base_dir=base_dir,
    )


def generate_timestamped_filename(chart_type: str, region: str, base_dir: str = "images") -> str:
    """
    Generate filename with current timestamp.

    Args:
        chart_type: Type of chart
        region: Region code
        base_dir: Base directory (relative to project root)

    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find project root and create absolute path
    project_root = _find_project_root()
    full_base_dir = os.path.join(project_root, base_dir)

    os.makedirs(full_base_dir, exist_ok=True)
    filename = f"{chart_type}_{region.lower()}_{timestamp}.png"

    return os.path.join(full_base_dir, filename)
