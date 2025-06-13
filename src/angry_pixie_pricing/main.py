"""Main CLI entry point for angry_pixie_pricing."""

import click
import numpy as np
from typing import Optional
from datetime import datetime
from .data.factory import DataSourceFactory
from .utils.date_parser import parse_date_range, format_date_range_description
from .utils.cli_options import (
    add_standard_options,
    add_chart_options,
    add_duck_factor_options,
    add_negative_pricing_options,
)
from .charts.terminal import (
    create_terminal_price_chart,
    create_terminal_price_summary,
    create_terminal_daily_average_chart,
    create_hourly_analysis_chart,
    create_hourly_workday_chart,
    create_png_price_chart,
    create_png_hourly_analysis_chart,
    create_png_hourly_workday_chart,
    create_terminal_duck_factor_chart,
    create_png_duck_factor_chart,
    create_png_seasonal_duck_chart,
    create_terminal_negative_pricing_chart,
    create_png_negative_pricing_chart,
    create_terminal_negative_pricing_timechart,
    create_png_negative_pricing_timechart,
)
from .analysis.rolling_duck import analyze_rolling_duck_patterns
from .analysis.negative_pricing import analyze_negative_pricing_comprehensive
from .utils.filename_generator import (
    generate_duck_factor_filename,
    generate_price_chart_filename,
    get_multi_window_filenames,
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
@add_standard_options
@add_chart_options
@click.pass_context
def chart(
    ctx,
    region: str,
    start_date: str,
    end_date: Optional[str],
    output: Optional[str],
    png: bool,
    no_cache: bool,
    chart_type: str,
    width: Optional[int],
    height: Optional[int],
):
    """Generate hourly electricity price charts for a region and time span."""
    try:
        # Parse dates with flexible format support
        start_dt, end_dt = parse_date_range(start_date, end_date)
        
        # Format dates for display and filename generation
        start_date_str = start_dt.strftime("%Y-%m-%d")
        end_date_str = end_dt.strftime("%Y-%m-%d")
        date_description = format_date_range_description(start_dt, end_dt)

        # Create data source
        data_source = DataSourceFactory.create_data_source(
            ctx.obj["data_source"], ctx.obj["cache_dir"]
        )

        click.echo(
            f"Fetching price data for {region} from {date_description}..."
        )

        # Get price data
        df = data_source.get_spot_prices(
            region, start_dt, end_dt, use_cache=not no_cache
        )

        click.echo(f"Retrieved {len(df)} hourly price points")
        click.echo(
            f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f} {df['unit'].iloc[0]}"
        )

        # Generate charts based on chart_type and output format
        if png or output:
            # PNG output mode - use smart filename generation if no extension provided
            if png:
                # Auto-generate filename when using --png flag
                output = generate_price_chart_filename(region, start_date_str, end_date_str, chart_type)
            elif not output.lower().endswith('.png'):
                # Add .png extension if not present
                output += '.png'
            
            try:
                if chart_type == "line":
                    create_png_price_chart(df, region, output, width=width or 12, height=height or 6)
                elif chart_type == "hourly":
                    create_png_hourly_analysis_chart(df, region, output, width=width or 12, height=height or 6)
                elif chart_type == "hourly-workday":
                    create_png_hourly_workday_chart(df, region, output, width=width or 12, height=height or 6)
                elif chart_type == "all":
                    # Create multiple PNG files with smart names
                    line_output = generate_price_chart_filename(region, start_date_str, end_date_str, "line")
                    hourly_output = generate_price_chart_filename(region, start_date_str, end_date_str, "hourly")
                    workday_output = generate_price_chart_filename(region, start_date_str, end_date_str, "hourly-workday")
                    
                    create_png_price_chart(df, region, line_output, width=width or 12, height=height or 6)
                    create_png_hourly_analysis_chart(df, region, hourly_output, width=width or 12, height=height or 6)
                    create_png_hourly_workday_chart(df, region, workday_output, width=width or 12, height=height or 6)
                else:
                    click.echo(f"PNG output not supported for chart type: {chart_type}")
                    click.echo("Supported PNG chart types: line, hourly, hourly-workday, all")
                    ctx.exit(1)
            except ImportError as e:
                click.echo(f"Error: {e}", err=True)
                click.echo("Please install matplotlib: pip install matplotlib>=3.7.0")
                ctx.exit(1)
        else:
            # Terminal output mode (default)
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
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD, YYYY-MM, or YYYY)")
@click.option("--end-date", help="End date (YYYY-MM-DD, YYYY-MM, or YYYY) - defaults to today")
@click.option("--no-cache", is_flag=True, help="Skip cache and fetch fresh data")
@click.pass_context
def calculate(
    ctx, usage_data: str, region: str, start_date: str, end_date: Optional[str], no_cache: bool
):
    """Calculate average electricity costs based on smart meter usage data."""
    try:
        # Parse dates with flexible format support
        start_dt, end_dt = parse_date_range(start_date, end_date)
        date_description = format_date_range_description(start_dt, end_dt)

        # Create data source
        data_source = DataSourceFactory.create_data_source(
            ctx.obj["data_source"], ctx.obj["cache_dir"]
        )

        click.echo(f"Calculating costs for {region} using {usage_data}")
        click.echo(f"Fetching price data from {date_description}...")

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


@cli.command()
@add_standard_options
@add_duck_factor_options
@click.pass_context
def duck_factor(
    ctx,
    region: str,
    start_date: str,
    end_date: Optional[str],
    window: str,
    step: str,
    output: Optional[str],
    png: bool,
    chart_type: str,
    no_cache: bool,
    width: Optional[int],
    height: Optional[int],
):
    """Analyze duck curve evolution over time with rolling window analysis."""
    try:
        # Parse dates with flexible format support
        start_dt, end_dt = parse_date_range(start_date, end_date)
        
        # Format dates for display and filename generation
        start_date_str = start_dt.strftime("%Y-%m-%d")
        end_date_str = end_dt.strftime("%Y-%m-%d")
        date_description = format_date_range_description(start_dt, end_dt)
        
        # Parse window and step sizes
        window_days = _parse_time_period(window)
        step_days = _parse_time_period(step)
        
        # Create data source
        data_source = DataSourceFactory.create_data_source(
            ctx.obj["data_source"], ctx.obj["cache_dir"]
        )
        
        click.echo(f"Fetching price data for {region} from {date_description}...")
        click.echo(f"Rolling window: {window_days} days, step: {step_days} days")
        
        # Get price data
        df = data_source.get_spot_prices(
            region, start_dt, end_dt, use_cache=not no_cache
        )
        
        click.echo(f"Retrieved {len(df)} hourly price points")
        
        # Perform rolling duck factor analysis
        click.echo("Analyzing duck curve evolution...")
        analysis_results = analyze_rolling_duck_patterns(df, region, window_days, step_days)
        
        if 'error' in analysis_results:
            click.echo(f"Analysis error: {analysis_results['error']}", err=True)
            ctx.exit(1)
        
        duck_factors = analysis_results['duck_factors']
        seasonal_data = analysis_results['seasonal_patterns']
        trends = analysis_results['trends']
        yoy_data = analysis_results['year_over_year']
        
        click.echo(f"Generated {len(duck_factors)} duck factor data points")
        
        # Display analysis summary
        _display_duck_factor_summary(duck_factors, trends, seasonal_data, yoy_data)
        
        # Generate charts
        if png or output:
            # PNG output mode - use smart filename generation
            try:
                if chart_type == "time-series":
                    if png:
                        output = generate_duck_factor_filename(region, start_date_str, end_date_str, window_days, "timeseries")
                    elif not output.lower().endswith('.png'):
                        output += '.png'
                    create_png_duck_factor_chart(duck_factors, region, output, window_days, 
                                                width=width or 12, height=height or 6)
                elif chart_type == "seasonal":
                    if png:
                        output = generate_duck_factor_filename(region, start_date_str, end_date_str, window_days, "seasonal")
                    elif not output.lower().endswith('.png'):
                        output += '.png'
                    create_png_seasonal_duck_chart(seasonal_data, region, output,
                                                  width=width or 12, height=height or 8)
                elif chart_type == "multi-window":
                    # Create multiple window analysis with smart filenames
                    from .analysis.rolling_duck import RollingDuckAnalyzer
                    analyzer = RollingDuckAnalyzer(region)
                    multi_results = analyzer.multi_window_analysis(df, [7, 30, 90], step_days)
                    
                    window_filenames = get_multi_window_filenames(region, start_date_str, end_date_str, [7, 30, 90])
                    
                    for window_key, window_df in multi_results.items():
                        window_output = window_filenames[window_key]
                        create_png_duck_factor_chart(window_df, region, window_output, 
                                                    int(window_key[:-1]), width=width or 12, height=height or 6)
                elif chart_type == "all":
                    # Create all chart types with smart filenames
                    timeseries_output = generate_duck_factor_filename(region, start_date_str, end_date_str, window_days, "timeseries")
                    seasonal_output = generate_duck_factor_filename(region, start_date_str, end_date_str, window_days, "seasonal")
                    
                    create_png_duck_factor_chart(duck_factors, region, timeseries_output, 
                                                window_days, width=width or 12, height=height or 6)
                    create_png_seasonal_duck_chart(seasonal_data, region, seasonal_output,
                                                  width=width or 12, height=height or 8)
                    
            except ImportError as e:
                click.echo(f"Error: {e}", err=True)
                click.echo("Please install matplotlib: pip install matplotlib>=3.7.0")
                ctx.exit(1)
        else:
            # Terminal output mode (default)
            if chart_type == "time-series" or chart_type == "all":
                create_terminal_duck_factor_chart(duck_factors, region, window_days, 
                                                 width=width, height=height)
            
            # Additional terminal analysis summaries for other chart types
            if chart_type == "seasonal" or chart_type == "all":
                _display_seasonal_analysis(seasonal_data, region)
            
            if chart_type == "multi-window" or chart_type == "all":
                _display_multi_window_analysis(df, region, step_days)
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        ctx.exit(1)


def _parse_time_period(period_str: str) -> int:
    """Parse time period string like '30d', '7d' into number of days."""
    period_str = period_str.lower().strip()
    
    if period_str.endswith('d'):
        return int(period_str[:-1])
    elif period_str.endswith('w'):
        return int(period_str[:-1]) * 7
    elif period_str.endswith('m'):
        return int(period_str[:-1]) * 30  # Approximate
    elif period_str.endswith('y'):
        return int(period_str[:-1]) * 365  # Approximate
    else:
        # Assume it's just a number of days
        return int(period_str)


def _display_duck_factor_summary(duck_factors, trends, seasonal_data, yoy_data):
    """Display a summary of duck factor analysis results."""
    click.echo("\n" + "="*60)
    click.echo("DUCK FACTOR ANALYSIS SUMMARY")
    click.echo("="*60)
    
    if not duck_factors.empty:
        avg_factor = duck_factors['duck_factor'].mean()
        min_factor = duck_factors['duck_factor'].min()
        max_factor = duck_factors['duck_factor'].max()
        
        click.echo(f"Average Duck Factor: {avg_factor:.3f}")
        click.echo(f"Range: {min_factor:.3f} - {max_factor:.3f}")
        
        # Trend information
        if 'trend_slope_per_year' in trends:
            slope = trends['trend_slope_per_year']
            r_squared = trends.get('trend_r_squared', 0)
            classification = trends.get('trend_classification', 'Unknown')
            
            click.echo(f"Trend: {classification} ({slope:+.4f}/year, R²={r_squared:.3f})")
        
        # Seasonal information
        if seasonal_data and 'peak_season' in seasonal_data:
            peak_season = seasonal_data['peak_season']
            low_season = seasonal_data['low_season']
            seasonal_range = seasonal_data.get('seasonal_range', 0)
            
            click.echo(f"Seasonal Pattern: Peak in {peak_season}, Low in {low_season}")
            click.echo(f"Seasonal Range: {seasonal_range:.3f}")
        
        # Year-over-year
        if yoy_data and 'avg_annual_change' in yoy_data:
            avg_change = yoy_data['avg_annual_change']
            click.echo(f"Average Annual Change: {avg_change:+.1%}")
    
    click.echo("="*60)


def _display_seasonal_analysis(seasonal_data, region):
    """Display seasonal analysis in terminal format."""
    if not seasonal_data or seasonal_data.get('seasonal_patterns', pd.DataFrame()).empty:
        click.echo("No seasonal data available")
        return
    
    click.echo(f"\n{'='*50}")
    click.echo(f"SEASONAL DUCK FACTOR ANALYSIS - {_get_country_name(region)}")
    click.echo(f"{'='*50}")
    
    seasonal_df = seasonal_data['seasonal_patterns']
    for _, row in seasonal_df.iterrows():
        click.echo(f"{row['season']:<8}: {row['mean']:.3f} ±{row['std']:.3f} ({row['count']} samples)")
    
    if 'peak_season' in seasonal_data:
        click.echo(f"\nPeak Season: {seasonal_data['peak_season']}")
        click.echo(f"Low Season: {seasonal_data['low_season']}")
        click.echo(f"Seasonal Range: {seasonal_data.get('seasonal_range', 0):.3f}")


def _display_multi_window_analysis(df, region, step_days):
    """Display multi-window analysis summary."""
    from .analysis.rolling_duck import RollingDuckAnalyzer
    
    analyzer = RollingDuckAnalyzer(region)
    multi_results = analyzer.multi_window_analysis(df, [7, 30, 90], step_days)
    
    click.echo(f"\n{'='*50}")
    click.echo(f"MULTI-WINDOW DUCK FACTOR COMPARISON")
    click.echo(f"{'='*50}")
    
    for window_key, window_df in multi_results.items():
        if not window_df.empty:
            avg_factor = window_df['duck_factor'].mean()
            volatility = window_df['duck_factor'].std()
            click.echo(f"{window_key:<8}: avg={avg_factor:.3f}, volatility={volatility:.3f}")


@cli.command()
@add_standard_options
@add_negative_pricing_options
@click.pass_context
def negative_pricing(
    ctx,
    region: str,
    start_date: str,
    end_date: Optional[str],
    threshold: float,
    chart_type: str,
    output: Optional[str],
    png: bool,
    no_cache: bool,
    width: Optional[int],
    height: Optional[int],
):
    """Analyze negative and near-zero electricity pricing patterns with solar potential estimates."""
    try:
        # Parse dates with flexible format support
        start_dt, end_dt = parse_date_range(start_date, end_date)
        
        # Format dates for display and filename generation
        start_date_str = start_dt.strftime("%Y-%m-%d")
        end_date_str = end_dt.strftime("%Y-%m-%d")
        date_description = format_date_range_description(start_dt, end_dt)
        
        # Create data source
        data_source = DataSourceFactory.create_data_source(
            ctx.obj["data_source"], ctx.obj["cache_dir"]
        )
        
        click.echo(f"Fetching price data for {region} from {date_description}...")
        click.echo(f"Near-zero threshold: {threshold} EUR/MWh")
        
        # Get price data
        df = data_source.get_spot_prices(
            region, start_dt, end_dt, use_cache=not no_cache
        )
        
        click.echo(f"Retrieved {len(df)} hourly price points")
        
        # Handle different chart types
        if chart_type == "timechart":
            # Timechart mode - no need for comprehensive analysis
            click.echo("Generating daily hours timechart...")
            
            # Handle PNG output - either via --png flag or --output
            if png or output:
                # PNG output mode
                if png:
                    # Auto-generate filename when using --png flag
                    output = f"images/negative-pricing-timechart_{region.lower()}_{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}.png"
                elif not output.lower().endswith('.png'):
                    # Add .png extension if not present
                    output += '.png'
                
                try:
                    create_png_negative_pricing_timechart(df, region, output, width=width or 12, height=height or 6, near_zero_threshold=threshold)
                except ImportError as e:
                    click.echo(f"Error: {e}", err=True)
                    click.echo("Please install matplotlib: pip install matplotlib>=3.7.0")
                    ctx.exit(1)
            else:
                # Terminal output mode (default)
                create_terminal_negative_pricing_timechart(df, region, width=width, height=height, near_zero_threshold=threshold)
        else:
            # Analysis mode (default)
            click.echo("Analyzing negative pricing patterns...")
            analysis_results = analyze_negative_pricing_comprehensive(df, region, threshold)
            
            if 'error' in analysis_results:
                click.echo(f"Analysis error: {analysis_results['error']}", err=True)
                ctx.exit(1)
            
            metrics = analysis_results['overall_metrics']
            seasonal_data = analysis_results['seasonal_patterns']
            
            # Display analysis summary
            _display_negative_pricing_summary(metrics, seasonal_data, region)
            
            # Generate charts
            if png or output:
                # PNG output mode
                if png:
                    # Auto-generate filename when using --png flag
                    output = f"images/negative-pricing_{region.lower()}_{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}.png"
                elif not output.lower().endswith('.png'):
                    # Add .png extension if not present
                    output += '.png'
                
                try:
                    create_png_negative_pricing_chart(df, region, output, width=width or 12, height=height or 8, near_zero_threshold=threshold)
                except ImportError as e:
                    click.echo(f"Error: {e}", err=True)
                    click.echo("Please install matplotlib: pip install matplotlib>=3.7.0")
                    ctx.exit(1)
            else:
                # Terminal output mode (default)
                create_terminal_negative_pricing_chart(df, region, width=width, height=height, near_zero_threshold=threshold)
        
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        ctx.exit(1)


def _display_negative_pricing_summary(metrics, seasonal_data, region):
    """Display comprehensive negative pricing analysis summary."""
    click.echo("\n" + "="*70)
    click.echo("NEGATIVE PRICING ANALYSIS SUMMARY")
    click.echo("="*70)
    
    # Overall statistics
    click.echo(f"Total negative hours: {metrics.negative_hours} ({metrics.negative_percentage:.1f}%)")
    click.echo(f"Total near-zero hours: {metrics.near_zero_hours} ({metrics.near_zero_percentage:.1f}%)")
    click.echo(f"Average negative hours per day: {metrics.avg_hours_per_day:.1f}")
    click.echo(f"Maximum consecutive negative hours: {metrics.max_consecutive_hours}")
    
    # Peak negative pricing hours
    if metrics.hourly_breakdown:
        peak_hour = max(metrics.hourly_breakdown.keys(), 
                       key=lambda h: metrics.hourly_breakdown[h].get('negative_percentage', 0))
        peak_percentage = metrics.hourly_breakdown[peak_hour]['negative_percentage']
        click.echo(f"Peak negative pricing hour: {peak_hour}:00 ({peak_percentage:.1f}% of the time)")
    
    # Monthly insights
    if metrics.monthly_breakdown:
        monthly_avg = [(month, data['avg_hours_per_day']) for month, data in metrics.monthly_breakdown.items()]
        monthly_avg.sort(key=lambda x: x[1], reverse=True)
        
        best_month = monthly_avg[0]
        worst_month = monthly_avg[-1]
        
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        click.echo(f"Best month for negative pricing: {month_names[best_month[0]]} ({best_month[1]:.1f} hours/day)")
        click.echo(f"Worst month for negative pricing: {month_names[worst_month[0]]} ({worst_month[1]:.1f} hours/day)")
    
    # Solar potential analysis
    if 'yearly_summary' in seasonal_data:
        yearly = seasonal_data['yearly_summary']
        click.echo(f"\nSOLAR POTENTIAL ANALYSIS (based on approximate solar irradiation data):")
        click.echo(f"Current average: {yearly['avg_current_hours_per_day']:.1f} hours/day")
        click.echo(f"Theoretical maximum: {yearly['avg_theoretical_max_hours_per_day']:.1f} hours/day")
        click.echo(f"Overall progress: {yearly['overall_progress_percentage']:.1f}%")
        
        remaining = yearly['avg_theoretical_max_hours_per_day'] - yearly['avg_current_hours_per_day']
        click.echo(f"Remaining potential: {remaining:.1f} hours/day")
        click.echo(f"NOTE: Solar potential estimates based on EU PVGIS, Global Solar Atlas, and Copernicus data")
    
    # Regional insights
    click.echo(f"\nREGIONAL CONTEXT ({_get_country_name(region)}):")
    
    # Show seasonal variation
    if len(seasonal_data) > 1:  # More than just yearly_summary
        summer_months = [6, 7, 8]  # June, July, August
        winter_months = [12, 1, 2]  # Dec, Jan, Feb
        
        summer_values = [seasonal_data[m]['progress_metrics']['current_hours_per_day'] 
                        for m in summer_months if m in seasonal_data and 'error' not in seasonal_data[m]['progress_metrics']]
        winter_values = [seasonal_data[m]['progress_metrics']['current_hours_per_day'] 
                        for m in winter_months if m in seasonal_data and 'error' not in seasonal_data[m]['progress_metrics']]
        
        summer_avg = np.mean(summer_values) if summer_values else np.nan
        winter_avg = np.mean(winter_values) if winter_values else np.nan
        
        if not np.isnan(summer_avg) and not np.isnan(winter_avg):
            seasonal_ratio = summer_avg / winter_avg if winter_avg > 0 else float('inf')
            click.echo(f"Summer vs Winter ratio: {seasonal_ratio:.1f}x (Summer: {summer_avg:.1f}h, Winter: {winter_avg:.1f}h)")
    
    click.echo("="*70)


def _get_country_name(region_code: str) -> str:
    """Get country name from region code."""
    country_names = {
        'AT': 'Austria', 'BE': 'Belgium', 'CH': 'Switzerland', 'CZ': 'Czech Republic',
        'DE': 'Germany', 'DK': 'Denmark', 'ES': 'Spain', 'FR': 'France',
        'IT': 'Italy', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland',
        'SE': 'Sweden', 'UK': 'United Kingdom'
    }
    return country_names.get(region_code.upper(), region_code)


if __name__ == "__main__":
    cli()
