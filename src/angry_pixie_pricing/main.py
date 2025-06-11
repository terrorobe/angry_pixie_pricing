"""Main CLI entry point for angry_pixie_pricing."""

import click
from typing import Optional
from datetime import datetime
from .data.factory import DataSourceFactory
from .charts.terminal import (
    create_terminal_price_chart,
    create_terminal_price_summary,
    create_terminal_daily_average_chart,
    create_hourly_analysis_chart,
    create_hourly_workday_chart,
)


@click.group()
@click.version_option()
@click.option(
    "--data-source", default="default", help="Data source to use (energy-charts)"
)
@click.option("--cache-dir", help="Cache directory for storing fetched data")
@click.pass_context
def cli(ctx, data_source: str, cache_dir: Optional[str]):
    """Angry Pixie Pricing - European Electricity Price Analysis Tool."""
    # Store common options in context
    ctx.ensure_object(dict)
    ctx.obj["data_source"] = data_source
    ctx.obj["cache_dir"] = cache_dir


@cli.command()
@click.option("--region", required=True, help="European region code (e.g., DE, FR, NL)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--output", "-o", help="Output file path for chart")
@click.option("--no-cache", is_flag=True, help="Skip cache and fetch fresh data")
@click.option(
    "--chart-type",
    "-t",
    type=click.Choice(["line", "daily", "summary", "hourly", "hourly-workday", "all"]),
    default="line",
    help="Type of chart to display",
)
@click.option("--width", type=int, help="Chart width (terminal columns)")
@click.option("--height", type=int, help="Chart height (terminal rows)")
@click.pass_context
def chart(
    ctx,
    region: str,
    start_date: str,
    end_date: str,
    output: Optional[str],
    no_cache: bool,
    chart_type: str,
    width: Optional[int],
    height: Optional[int],
):
    """Generate hourly electricity price charts for a region and time span."""
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Create data source
        data_source = DataSourceFactory.create_data_source(
            ctx.obj["data_source"], ctx.obj["cache_dir"]
        )

        click.echo(
            f"Fetching price data for {region} from {start_date} to {end_date}..."
        )

        # Get price data
        df = data_source.get_spot_prices(
            region, start_dt, end_dt, use_cache=not no_cache
        )

        click.echo(f"Retrieved {len(df)} hourly price points")
        click.echo(
            f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f} {df['unit'].iloc[0]}"
        )

        # Generate charts based on chart_type
        if output:
            click.echo(f"Note: File output not yet implemented, displaying in terminal")

        if chart_type == "line" or chart_type == "all":
            create_terminal_price_chart(df, region, width=width, height=height)

        if chart_type == "daily" or chart_type == "all":
            create_terminal_daily_average_chart(df, region)

        if chart_type == "summary" or chart_type == "all":
            create_terminal_price_summary(df, region)
        
        if chart_type == "hourly":
            create_hourly_analysis_chart(df, region)
        
        if chart_type == "hourly-workday":
            create_hourly_workday_chart(df, region)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option("--usage-data", required=True, help="Path to smart meter usage data file")
@click.option("--region", required=True, help="European region code (e.g., DE, FR, NL)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--no-cache", is_flag=True, help="Skip cache and fetch fresh data")
@click.pass_context
def calculate(
    ctx, usage_data: str, region: str, start_date: str, end_date: str, no_cache: bool
):
    """Calculate average electricity costs based on smart meter usage data."""
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Create data source
        data_source = DataSourceFactory.create_data_source(
            ctx.obj["data_source"], ctx.obj["cache_dir"]
        )

        click.echo(f"Calculating costs for {region} using {usage_data}")
        click.echo(f"Fetching price data from {start_date} to {end_date}...")

        # Get price data
        df = data_source.get_spot_prices(
            region, start_dt, end_dt, use_cache=not no_cache
        )

        click.echo(f"Retrieved {len(df)} hourly price points")

        # TODO: Implement cost calculation logic with smart meter data
        click.echo("Cost calculation would be performed here")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        ctx.exit(1)


@cli.command()
def sources():
    """List available data sources."""
    sources = DataSourceFactory.list_available_sources()
    click.echo("Available data sources:")
    for source in sources:
        ds = DataSourceFactory.create_data_source(source)
        info = ds.get_data_source_info()
        click.echo(f"  {source}: {info['description']}")


@cli.command()
@click.option("--region", help="Clear cache for specific region only")
@click.pass_context
def clear_cache(ctx, region: Optional[str]):
    """Clear cached price data."""
    data_source = DataSourceFactory.create_data_source(
        ctx.obj["data_source"], ctx.obj["cache_dir"]
    )

    data_source.clear_cache(region)

    if region:
        click.echo(f"Cleared cache for region: {region}")
    else:
        click.echo("Cleared all cached data")


if __name__ == "__main__":
    cli()
