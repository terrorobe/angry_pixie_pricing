"""Shared CLI option definitions for reuse across commands."""

import click


# Common date options used across multiple commands
def add_date_options(func):
    """Decorator to add standard date options to a command."""
    func = click.option(
        "--start-date", required=True, help="Start date (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )(func)
    func = click.option(
        "--end-date", help="End date (YYYY-MM-DD, YYYY-MM, or YYYY) - defaults to today"
    )(func)
    return func


def add_region_option(func):
    """Decorator to add region option to a command."""
    return click.option("--region", required=True, help="European region code (e.g., DE, FR, NL)")(
        func
    )


def add_output_options(func):
    """Decorator to add output-related options to a command."""
    func = click.option(
        "--output", "-o", default=None, help="Output PNG file path (enables PNG mode)."
    )(func)
    func = click.option("--png", is_flag=True, help="Generate PNG with auto-generated filename.")(
        func
    )
    func = click.option("--width", type=int, help="Chart width (terminal columns or PNG inches)")(
        func
    )
    func = click.option("--height", type=int, help="Chart height (terminal rows or PNG inches)")(
        func
    )
    return func


def add_cache_option(func):
    """Decorator to add cache option to a command."""
    return click.option("--no-cache", is_flag=True, help="Skip cache and fetch fresh data")(func)


def add_standard_options(func):
    """Decorator to add the most common options (region, dates, output, cache)."""
    func = add_cache_option(func)
    func = add_output_options(func)
    func = add_date_options(func)
    func = add_region_option(func)
    return func


# Specialized option groups for specific commands


def add_duck_factor_options(func):
    """Decorator to add duck factor specific options."""
    func = click.option(
        "--window", "-w", default="30d", help="Rolling window size (e.g., 7d, 30d, 90d)"
    )(func)
    func = click.option(
        "--step", "-s", default="7d", help="Step size between calculations (e.g., 1d, 7d)"
    )(func)
    func = click.option(
        "--chart-type",
        "-t",
        type=click.Choice(["time-series", "seasonal", "multi-window", "all"]),
        default="time-series",
        help="Chart type: time-series, seasonal, multi-window, or all",
    )(func)
    return func


def add_chart_options(func):
    """Decorator to add chart command specific options."""
    func = click.option(
        "--chart-type",
        "-t",
        type=click.Choice(["line", "daily", "summary", "hourly", "hourly-workday", "all"]),
        default="line",
        help="""\
Chart type options:

\b
line: Hourly price timeline with braille markers
daily: Bar chart of daily average prices
summary: Statistical summary with price ranges
hourly: Duck curve analysis (workdays vs weekends)
hourly-workday: Workday-only duck curve pattern
all: Display line, daily, and summary together
""",
    )(func)
    return func


def add_negative_pricing_options(func):
    """Decorator to add negative pricing specific options."""
    func = click.option("--threshold", default=10.0, help="Near-zero price threshold (EUR/MWh)")(
        func
    )
    func = click.option(
        "--severe-threshold",
        default=-50.0,
        help="Severe negative price threshold (EUR/MWh, default: -50)",
    )(func)
    func = click.option(
        "--extreme-threshold",
        default=-100.0,
        help="Extreme negative price threshold (EUR/MWh, default: -100)",
    )(func)
    func = click.option(
        "--cheap-threshold", default=40.0, help="Cheap price threshold (EUR/MWh, default: 40)"
    )(func)
    func = click.option(
        "--chart-type",
        "-t",
        type=click.Choice(["analysis", "timechart"]),
        default="analysis",
        help="""\
Chart type options:

\b
analysis: Comprehensive 4-panel analysis (default)
timechart: Time series of daily hours with negative/near-zero prices
""",
    )(func)
    func = click.option(
        "--aggregation-level",
        "-a",
        type=click.Choice(["daily", "weekly", "monthly", "solar-quarters"]),
        default="daily",
        help="""\
Aggregation level for timechart (only applies to chart-type=timechart):

\b
daily: Hours per day (default)
weekly: Hours per week
monthly: Hours per month
solar-quarters: Hours per solar quarter (Peak Sun: May-Jul, Rising Sun: Feb-Apr, Fading Sun: Aug-Oct, Low Sun: Nov-Jan)
""",
    )(func)
    return func


# Alternative approach: Option classes for more complex scenarios
class CommonOptions:
    """Class to hold common option definitions that can be reused."""

    REGION = click.option("--region", required=True, help="European region code (e.g., DE, FR, NL)")

    START_DATE = click.option(
        "--start-date", required=True, help="Start date (YYYY-MM-DD, YYYY-MM, or YYYY)"
    )

    END_DATE = click.option(
        "--end-date", help="End date (YYYY-MM-DD, YYYY-MM, or YYYY) - defaults to today"
    )

    OUTPUT = click.option("--output", "-o", help="Output PNG file path (enables PNG mode)")

    NO_CACHE = click.option("--no-cache", is_flag=True, help="Skip cache and fetch fresh data")

    WIDTH = click.option("--width", type=int, help="Chart width (terminal columns or PNG inches)")

    HEIGHT = click.option("--height", type=int, help="Chart height (terminal rows or PNG inches)")


# For even more complex cases, you can use click.Group with common parameters
class CommonParameterGroup(click.Group):
    """Custom Click Group that adds common parameters to all commands."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def command(self, *args, **kwargs):
        """Override command decorator to add common options."""

        def decorator(f):
            # Add common options to every command
            f = add_cache_option(f)
            # Create the command normally
            cmd = super(CommonParameterGroup, self).command(*args, **kwargs)(f)
            return cmd

        return decorator


# Usage examples for documentation:
"""
USAGE EXAMPLES:

# Method 1: Decorators (recommended for most cases)
@cli.command()
@add_standard_options
def my_command(region, start_date, end_date, output, no_cache, width, height):
    pass

# Method 2: Individual decorators for fine control
@cli.command()
@add_region_option
@add_date_options
@add_duck_factor_options
def duck_factor_command(...):
    pass

# Method 3: Class-based options for complex cases
@cli.command()
@CommonOptions.REGION
@CommonOptions.START_DATE
@CommonOptions.END_DATE
def another_command(region, start_date, end_date):
    pass

# Method 4: Custom group (for when you want ALL commands to share options)
@click.group(cls=CommonParameterGroup)
def cli():
    pass
"""
