"""Main CLI entry point for angry_pixie_pricing."""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd

from .analysis.negative_pricing import analyze_negative_pricing_comprehensive
from .analysis.rolling_duck import analyze_rolling_duck_patterns
from .charts.terminal import (
    create_hourly_analysis_chart,
    create_hourly_workday_chart,
    create_png_duck_factor_chart,
    create_png_hourly_analysis_chart,
    create_png_hourly_workday_chart,
    create_png_negative_pricing_chart,
    create_png_negative_pricing_timechart,
    create_png_price_chart,
    create_png_seasonal_duck_chart,
    create_terminal_daily_average_chart,
    create_terminal_duck_factor_chart,
    create_terminal_negative_pricing_chart,
    create_terminal_negative_pricing_timechart,
    create_terminal_price_chart,
    create_terminal_price_summary,
)
from .data.factory import DataSourceFactory
from .utils.cli_options import (
    add_chart_options,
    add_duck_factor_options,
    add_negative_pricing_options,
    add_standard_options,
)
from .utils.date_parser import format_date_range_description, parse_date_range
from .utils.filename_generator import (
    generate_duck_factor_filename,
    generate_load_peaks_filename,
    generate_price_chart_filename,
    get_multi_window_filenames,
)


def _open_file(file_path: str) -> None:
    """Open a file using the system default application."""
    try:
        subprocess.run(["open", file_path], check=True)  # noqa: S603, S607
    except subprocess.CalledProcessError:
        click.echo(f"Failed to open {file_path}")
    except FileNotFoundError:
        # Fallback for non-macOS systems
        try:
            subprocess.run(["xdg-open", file_path], check=True)  # noqa: S603, S607
        except (subprocess.CalledProcessError, FileNotFoundError):
            click.echo(f"Could not open {file_path} - please open manually")


@click.group()
@click.version_option()
@click.option("--data-source", default="default", help="Data source to use (energy-charts)")
@click.option("--cache-dir", help="Cache directory for storing fetched data")
@click.pass_context
def cli(ctx: click.Context, data_source: str, cache_dir: str | None) -> None:
    """Angry Pixie Pricing - European Electricity Price Analysis Tool."""
    # Store common options in context
    ctx.ensure_object(dict)
    ctx.obj["data_source"] = data_source
    ctx.obj["cache_dir"] = cache_dir


@cli.command()
@add_standard_options
@add_chart_options
@click.option("--open", "open_file", is_flag=True, help="Open generated PNG file after creation")
@click.pass_context
def chart(
    ctx: click.Context,
    region: str,
    start_date: str,
    end_date: str | None,
    output: str | None,
    png: bool,
    no_cache: bool,
    chart_type: str,
    width: int | None,
    height: int | None,
    open_file: bool,
) -> None:
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
            ctx.obj["data_source"],
            ctx.obj["cache_dir"],
        )

        click.echo(f"Fetching price data for {region} from {date_description}...")

        # Get price data
        df = data_source.get_spot_prices(region, start_dt, end_dt, use_cache=not no_cache)

        click.echo(f"Retrieved {len(df)} hourly price points")
        click.echo(
            f"Price range: {df['price'].min():.2f} - {df['price'].max():.2f} {df['unit'].iloc[0]}",
        )

        # Generate charts based on chart_type and output format
        if png or output:
            # PNG output mode - use smart filename generation if no extension provided
            if png:
                # Auto-generate filename when using --png flag
                output = generate_price_chart_filename(
                    region,
                    start_date_str,
                    end_date_str,
                    chart_type,
                )
            elif output is not None and not output.lower().endswith(".png"):
                # Add .png extension if not present
                output += ".png"

            try:
                if chart_type == "line":
                    if output is not None:
                        create_png_price_chart(
                            df,
                            region,
                            output,
                            width=width or 12,
                            height=height or 6,
                        )
                        if open_file:
                            _open_file(output)
                elif chart_type == "hourly":
                    if output is not None:
                        create_png_hourly_analysis_chart(
                            df,
                            region,
                            output,
                            width=width or 12,
                            height=height or 6,
                        )
                        if open_file:
                            _open_file(output)
                elif chart_type == "hourly-workday":
                    if output is not None:
                        create_png_hourly_workday_chart(
                            df,
                            region,
                            output,
                            width=width or 12,
                            height=height or 6,
                        )
                        if open_file:
                            _open_file(output)
                elif chart_type == "all":
                    # Create multiple PNG files with smart names
                    line_output = generate_price_chart_filename(
                        region,
                        start_date_str,
                        end_date_str,
                        "line",
                    )
                    hourly_output = generate_price_chart_filename(
                        region,
                        start_date_str,
                        end_date_str,
                        "hourly",
                    )
                    workday_output = generate_price_chart_filename(
                        region,
                        start_date_str,
                        end_date_str,
                        "hourly-workday",
                    )

                    create_png_price_chart(
                        df,
                        region,
                        line_output,
                        width=width or 12,
                        height=height or 6,
                    )
                    create_png_hourly_analysis_chart(
                        df,
                        region,
                        hourly_output,
                        width=width or 12,
                        height=height or 6,
                    )
                    create_png_hourly_workday_chart(
                        df,
                        region,
                        workday_output,
                        width=width or 12,
                        height=height or 6,
                    )

                    if open_file:
                        _open_file(line_output)
                        _open_file(hourly_output)
                        _open_file(workday_output)
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
    except OSError as e:
        click.echo(f"File I/O error: {e}", err=True)
        ctx.exit(1)
    except (ConnectionError, TimeoutError) as e:
        click.echo(f"Network error while fetching data: {e}", err=True)
        ctx.exit(1)
    except pd.errors.EmptyDataError as e:
        click.echo(f"No data available for the specified period: {e}", err=True)
        ctx.exit(1)
    except KeyError as e:
        click.echo(f"Data format error - missing expected field: {e}", err=True)
        ctx.exit(1)


@cli.command()
@click.option("--usage-data", help="Path to smart meter usage data file (CSV)")
@click.option("--billing-data", help="Path to billing data JSON or use --peak-kw and --total-kwh")
@click.option("--peak-kw", type=float, help="Peak power in kW (for billing reconstruction)")
@click.option(
    "--total-kwh",
    type=float,
    help="Total consumption in kWh (for model calibration period)",
)
@click.option("--days", type=int, help="Billing period length in days (for model calibration)")
@click.option(
    "--daily-kwh",
    type=float,
    help="Average daily consumption in kWh (alternative to --total-kwh --days)",
)
@click.option(
    "--profile-type",
    type=click.Choice(["residential", "commercial", "industrial", "kids_hotel"]),
    default="residential",
    help="Building profile type for reconstruction",
)
@click.option(
    "--region",
    help="European region code (e.g., DE, FR, NL) - not needed with --profile-only",
)
@click.option(
    "--start-date",
    help="Start date (YYYY-MM-DD, YYYY-MM, or YYYY) - not needed with --profile-only",
)
@click.option("--end-date", help="End date (YYYY-MM-DD, YYYY-MM, or YYYY) - defaults to today")
@click.option("--compare-flat-rate", type=float, help="Compare with flat rate tariff (EUR/kWh)")
@click.option("--output", help="Save detailed results to CSV file")
@click.option("--no-cache", is_flag=True, help="Skip cache and fetch fresh data")
# Kids hotel specific options
@click.option(
    "--occupancy-rate",
    type=float,
    default=0.7,
    help="Hotel occupancy rate (0-1, default: 0.7)",
)
@click.option(
    "--kitchen-weight",
    type=float,
    default=0.2,
    help="Kitchen/restaurant load weight (default: 0.2)",
)
@click.option(
    "--wellness-weight",
    type=float,
    default=0.25,
    help="Pool/wellness load weight (default: 0.25)",
)
@click.option(
    "--activity-weight",
    type=float,
    default=0.15,
    help="Activity/game areas load weight (default: 0.15)",
)
@click.option(
    "--rooms-weight",
    type=float,
    default=0.25,
    help="Guest rooms load weight (default: 0.25)",
)
@click.option(
    "--common-weight",
    type=float,
    default=0.15,
    help="Common areas load weight (default: 0.15)",
)
@click.option(
    "--day-kwh",
    type=float,
    help="Day consumption kWh (06:00-22:00) for calibration period",
)
@click.option(
    "--night-kwh",
    type=float,
    help="Night consumption kWh (22:00-06:00) for calibration period",
)
@click.option("--show-profile", is_flag=True, help="Display load profile chart in terminal")
@click.option("--profile-chart", help="Save load profile chart to PNG file")
@click.option("--cost-chart", help="Save monthly cost breakdown chart to PNG file")
@click.option(
    "--handling-fee",
    type=float,
    default=0.015,
    help="Handling fee per kWh (default: 0.015 EUR/kWh = 1.5ct)",
)
@click.option(
    "--profile-only",
    is_flag=True,
    help="Generate profile only (no cost calculation, no dates needed)",
)
@click.option("--open", "open_file", is_flag=True, help="Open generated PNG files after creation")
@click.pass_context
def calculate(
    ctx: click.Context,
    usage_data: str | None,
    billing_data: str | None,
    peak_kw: float | None,
    total_kwh: float | None,
    days: int | None,
    daily_kwh: float | None,
    profile_type: str,
    region: str,
    start_date: str,
    end_date: str | None,
    compare_flat_rate: float | None,
    output: str | None,
    no_cache: bool,
    occupancy_rate: float,
    kitchen_weight: float,
    wellness_weight: float,
    activity_weight: float,
    rooms_weight: float,
    common_weight: float,
    day_kwh: float | None,
    night_kwh: float | None,
    show_profile: bool,
    profile_chart: str | None,
    cost_chart: str | None,
    handling_fee: float,
    profile_only: bool,
    open_file: bool,
) -> None:
    """Calculate electricity costs using smart meter data or billing reconstruction."""
    try:
        from .analysis.cost_calculator import CostCalculator
        from .load_profiles import (
            KidsHotelProfile,
            ProfileType,
            SmartMeterProfile,
        )
        from .load_profiles.calibrated import CalibratedProfile

        # Handle profile-only mode
        if profile_only:
            # Use reference day for profile generation
            start_dt = datetime(2024, 7, 15, 0, 0)  # Reference Monday
            end_dt = datetime(2024, 7, 15, 23, 59, 59)
            date_description = "Reference day (for profile pattern)"
            click.echo("Profile-only mode: generating daily pattern")
        else:
            # Parse dates with flexible format support
            if not start_date:
                msg = "--start-date is required unless using --profile-only"
                raise ValueError(msg)
            if not region:
                msg = "--region is required unless using --profile-only"
                raise ValueError(msg)
            start_dt, end_dt = parse_date_range(start_date, end_date)

            # Constrain end date to today if in the future
            today = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
            if end_dt > today:
                click.echo(f"Note: End date constrained to {today.strftime('%Y-%m-%d')} (today)")
                end_dt = today
            date_description = format_date_range_description(start_dt, end_dt)

        # Determine load profile source
        if usage_data:
            click.echo(f"Loading smart meter data from {usage_data}")
            load_profile = SmartMeterProfile.from_standard_formats(Path(usage_data))

            # Validate data quality
            validation = load_profile.validate_data()
            click.echo(f"Data quality score: {validation['data_quality_score']:.1f}%")

        elif billing_data or (peak_kw and (total_kwh or daily_kwh)):
            click.echo("Creating calibrated load profile from billing data")

            # Calculate daily consumption
            if daily_kwh:
                daily_consumption = daily_kwh
                click.echo(f"Using daily consumption: {daily_kwh:.1f} kWh/day")
            elif total_kwh and days:
                daily_consumption = total_kwh / days
                click.echo(
                    f"Calculated daily consumption: {total_kwh:.1f} kWh ÷ {days} days = "
                    f"{daily_consumption:.1f} kWh/day",
                )
            elif total_kwh:
                # Assume total_kwh is daily if no days specified
                daily_consumption = total_kwh
                click.echo(
                    f"Assuming daily consumption: {total_kwh:.1f} kWh/day (use --days if this is a monthly total)",
                )
            else:
                msg = "Must provide either --daily-kwh or --total-kwh (with optional --days)"
                raise ValueError(
                    msg,
                )

            if not peak_kw:
                msg = "Must provide --peak-kw for load profile calibration"
                raise ValueError(msg)

            # Create profile template
            if profile_type == "kids_hotel":
                # Validate facility weights sum to approximately 1.0
                total_weight = kitchen_weight + wellness_weight + activity_weight + rooms_weight + common_weight
                if abs(total_weight - 1.0) > 0.05:
                    click.echo(
                        f"Warning: Facility weights sum to {total_weight:.2f}, should be close to 1.0",
                    )

                facility_weights = {
                    "kitchen": kitchen_weight,
                    "wellness": wellness_weight,
                    "activity": activity_weight,
                    "rooms": rooms_weight,
                    "common": common_weight,
                }

                profile_template: Any = KidsHotelProfile(
                    occupancy_rate=occupancy_rate,
                    facility_weights=facility_weights,
                )

                click.echo(f"Kids hotel profile: {occupancy_rate:.0%} occupancy")
                click.echo(
                    f"Facility weights: Kitchen {kitchen_weight:.0%}, Pool {wellness_weight:.0%}, "
                    f"Activities {activity_weight:.0%}, Rooms {rooms_weight:.0%}, Common {common_weight:.0%}",
                )
            else:
                # Use standard profile types
                from .load_profiles.templates import get_standard_profile

                profile_type_enum = ProfileType(profile_type)
                profile_template = get_standard_profile(profile_type_enum)

            # Set up day/night split for calibration
            day_night_split = None
            if day_kwh is not None and night_kwh is not None:
                if days:
                    # Convert monthly to daily values
                    daily_day_kwh = day_kwh / days
                    daily_night_kwh = night_kwh / days
                else:
                    # Assume already daily values
                    daily_day_kwh = day_kwh
                    daily_night_kwh = night_kwh

                day_night_split = (daily_day_kwh, daily_night_kwh)
                click.echo(
                    f"Day/night split: {daily_day_kwh:.1f} kWh day, {daily_night_kwh:.1f} kWh night (daily)",
                )

                # Verify split matches total
                split_total = daily_day_kwh + daily_night_kwh
                if abs(split_total - daily_consumption) > 0.1:
                    click.echo(
                        f"Warning: Day/night split ({split_total:.1f} kWh) doesn't match "
                        f"daily total ({daily_consumption:.1f} kWh)",
                    )

            # Create calibrated profile
            calibrated_profile: Any = CalibratedProfile(
                calculation_start_date=start_dt,
                calculation_end_date=end_dt,
                daily_kwh=daily_consumption,
                peak_kw=peak_kw,
                profile_template=profile_template,
                day_night_split=day_night_split,
            )
            load_profile = calibrated_profile

            # Show calibration info
            calibration = calibrated_profile.get_calibration_info()
            click.echo(f"Model calibrated: {calibration['template_used']}")
            if "day_night_ratio" in calibration:
                click.echo(f"Day/night ratio: {calibration['day_night_ratio']}")
            click.echo(f"Calculating for {calibration['calculation_period_days']} days")
            click.echo(
                f"Target total consumption: {calibration['total_consumption_target']:.1f} kWh",
            )

        else:
            msg = "Either --usage-data or billing parameters must be provided"
            raise ValueError(msg)

        # Profile-only mode - skip cost calculations
        if profile_only:
            click.echo(f"\nGenerated profile pattern for {date_description}")

            # Show profile statistics
            stats = load_profile.get_statistics()
            click.echo("\n=== Profile Statistics ===")
            click.echo(f"Daily consumption: {stats['total_consumption_kwh']:.1f} kWh")
            click.echo(f"Peak power: {stats['peak_power_kw']:.1f} kW")
            click.echo(f"Average power: {stats['average_power_kw']:.1f} kW")
            click.echo(f"Load factor: {stats['load_factor']:.2f}")
            click.echo(f"Base load: {stats['base_load_kw']:.1f} kW")

            # Always show profile in profile-only mode
            _display_load_profile(load_profile, True, profile_chart, open_file)

            # Save data if requested
            if output:
                load_profile.data.to_csv(output)
                click.echo(f"\nProfile data saved to {output}")

            return  # Skip cost calculations

        # Regular mode - calculate costs
        click.echo(f"\nCalculating costs for {region} from {date_description}...")

        # Create simplified energy-only cost calculator
        calculator = CostCalculator(
            load_profile=load_profile,
            region=region,
            vat_rate=0,
            grid_fees_kwh=0,
            other_fees_kwh=0,
            fixed_monthly_fee=0,
        )

        # Calculate costs
        results = calculator.calculate(use_cache=not no_cache)

        # Display results with monthly aggregation
        try:
            detailed_df = calculator.get_detailed_breakdown()
            _display_monthly_results(
                detailed_df,
                results,
                start_dt,
                end_dt,
                cost_chart,
                peak_kw,
                daily_consumption if "daily_consumption" in locals() else None,
                day_kwh,
                night_kwh,
                occupancy_rate,
                open_file,
                region,
                handling_fee,
            )
        except (KeyError, ValueError, pd.errors.EmptyDataError) as e:
            # Fallback to simple display
            click.echo(f"Monthly aggregation failed: {e}")
            click.echo("\n=== Energy Costs ===")
            click.echo(f"Total consumption: {results['total_consumption_kwh']:.1f} kWh")
            click.echo(f"Energy cost: €{results['energy_cost']:.2f}")
            click.echo(f"Average price paid: €{results['weighted_avg_price_eur_kwh']:.4f}/kWh")

        # Compare with flat rate if requested
        if compare_flat_rate:
            # Simplified flat rate comparison (energy only)
            total_kwh = results["total_consumption_kwh"]
            flat_energy_cost = total_kwh * compare_flat_rate
            spot_energy_cost = results["energy_cost"]
            savings = flat_energy_cost - spot_energy_cost
            savings_percent = (savings / flat_energy_cost) * 100 if flat_energy_cost > 0 else 0

            click.echo("\n=== Flat Rate Comparison ===")
            click.echo(f"Flat rate cost: €{flat_energy_cost:.2f} (€{compare_flat_rate:.4f}/kWh)")
            click.echo(f"Spot market cost: €{spot_energy_cost:.2f}")
            click.echo(f"Savings: €{savings:.2f} ({savings_percent:.1f}%)")
            click.echo(f"Better option: {'spot_market' if savings > 0 else 'flat_rate'}")

        # Save detailed results if requested
        if output:
            detailed = calculator.get_detailed_breakdown()
            detailed.to_csv(output)
            click.echo(f"\nDetailed results saved to {output}")

        # Show load profile visualization
        if show_profile or profile_chart:
            _display_load_profile(load_profile, show_profile, profile_chart, open_file)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except (OSError, FileNotFoundError) as e:
        click.echo(f"File error: {e}", err=True)
        ctx.exit(1)
    except ImportError as e:
        click.echo(f"Missing dependency: {e}", err=True)
        ctx.exit(1)
    except (ConnectionError, TimeoutError) as e:
        click.echo(f"Network error while fetching data: {e}", err=True)
        ctx.exit(1)
    except pd.errors.ParserError as e:
        click.echo(f"Data parsing error: {e}", err=True)
        ctx.exit(1)
    except KeyError as e:
        click.echo(f"Data format error - missing expected field: {e}", err=True)
        ctx.exit(1)


def _display_load_profile(
    load_profile: Any,
    show_terminal: bool,
    save_png: str | None,
    open_file: bool = False,
) -> None:
    """Display load profile visualization."""
    import plotext as plt

    # Get profile data
    df = load_profile.data

    # For display, limit to reasonable time range
    if len(df) > 96:  # More than 1 day of 15-min data
        # Show first day only for terminal
        display_df = df.iloc[:96]  # First 24 hours
        time_desc = "First 24 hours"
    else:
        display_df = df
        hours = len(df) / 4
        time_desc = f"{hours:.0f} hours"

    if show_terminal:
        click.echo(f"\n=== Load Profile Chart ({time_desc}) ===")

        # Convert to hourly for better terminal display
        hourly_df = display_df.resample("h").mean()

        # Prepare data for plotext
        hour_indices: list[int] = list(range(len(hourly_df)))
        power_values: list[float] = hourly_df["power_kw"].tolist()

        # Create terminal chart
        plt.clear_data()
        plt.plot(hour_indices, power_values, marker="hd")

        plt.title("Hotel Load Profile")
        plt.xlabel("Hour of Day")
        plt.ylabel("Power (kW)")

        # Add statistics as text
        stats = load_profile.get_statistics()
        plt.text(
            f"Peak: {stats['peak_power_kw']:.1f} kW | "
            f"Avg: {stats['average_power_kw']:.1f} kW | "
            f"Load Factor: {stats['load_factor']:.2f}",
            x=0,
            y=max(power_values) * 0.9,
        )

        plt.show()

        # Show facility breakdown for kids hotel
        if hasattr(load_profile, "template") and hasattr(
            load_profile.template,
            "get_kitchen_load_factor",
        ):
            _show_facility_breakdown(load_profile.template, stats["peak_power_kw"])

    if save_png:
        _save_profile_png(load_profile, save_png)
        click.echo(f"\nLoad profile chart saved to {save_png}")
        if open_file:
            _open_file(save_png)


def _show_facility_breakdown(template: Any, peak_power: float) -> None:
    """Show facility load breakdown for kids hotel."""
    click.echo("\n=== Facility Load Breakdown (Peak Hour Estimates) ===")

    # Find peak hour (simplified - use hour 19 as typical evening peak)
    peak_hour = 19

    if hasattr(template, "facility_weights"):
        weights = template.facility_weights
        occupancy = template.occupancy_rate

        # Calculate estimated loads
        kitchen_load = template.get_kitchen_load_factor(peak_hour, occupancy) * weights["kitchen"] * peak_power
        wellness_load = template.get_wellness_load_factor(peak_hour, True) * weights["wellness"] * peak_power
        activity_load = template.get_activity_area_load(peak_hour) * weights["activity"] * peak_power
        room_load = template.get_guest_room_load(peak_hour, occupancy) * weights["rooms"] * peak_power
        common_load = template.get_common_area_load(peak_hour) * weights["common"] * peak_power

        click.echo(
            f"Kitchen/Restaurant: {kitchen_load:.1f} kW ({kitchen_load / peak_power * 100:.0f}%)",
        )
        click.echo(
            f"Pool/Wellness:      {wellness_load:.1f} kW ({wellness_load / peak_power * 100:.0f}%)",
        )
        click.echo(
            f"Activities:         {activity_load:.1f} kW ({activity_load / peak_power * 100:.0f}%)",
        )
        click.echo(f"Guest Rooms:        {room_load:.1f} kW ({room_load / peak_power * 100:.0f}%)")
        click.echo(
            f"Common Areas:       {common_load:.1f} kW ({common_load / peak_power * 100:.0f}%)",
        )
        click.echo(
            f"Total:              {kitchen_load + wellness_load + activity_load + room_load + common_load:.1f} kW",
        )


def _save_profile_png(load_profile: Any, filename: str) -> None:
    """Save load profile chart as PNG."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        import pandas as pd

        df = load_profile.data
        stats = load_profile.get_statistics()

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot load profile
        ax.plot(df.index, df["power_kw"], "b-", linewidth=1.5, alpha=0.8)

        # Add peak and average lines
        ax.axhline(
            y=stats["peak_power_kw"],
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Peak: {stats['peak_power_kw']:.1f} kW",
        )
        ax.axhline(
            y=stats["average_power_kw"],
            color="g",
            linestyle="--",
            alpha=0.7,
            label=f"Average: {stats['average_power_kw']:.1f} kW",
        )

        # Shade day/night periods if more than one day
        if len(df) > 96:  # More than 1 day
            for date in df.index.date:
                day_start = pd.Timestamp(date).replace(hour=6)
                day_end = pd.Timestamp(date).replace(hour=22)
                night_start = pd.Timestamp(date).replace(hour=22)
                night_end = pd.Timestamp(date + pd.Timedelta(days=1)).replace(hour=6)

                if day_start >= df.index[0] and day_end <= df.index[-1]:
                    ax.axvspan(
                        day_start,
                        day_end,
                        alpha=0.1,
                        color="yellow",
                        label="Day hours" if date == df.index.date[0] else "",
                    )
                if night_start >= df.index[0] and night_end <= df.index[-1]:
                    ax.axvspan(
                        night_start,
                        night_end,
                        alpha=0.1,
                        color="navy",
                        label="Night hours" if date == df.index.date[0] else "",
                    )

        # Formatting
        ax.set_title(
            f"Load Profile ({stats['start_date'].strftime('%Y-%m-%d')} to {stats['end_date'].strftime('%Y-%m-%d')})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Power (kW)")
        ax.set_ylim(bottom=0)  # Zero-index the power axis
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Format x-axis
        if len(df) <= 96:  # One day or less
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # type: ignore[no-untyped-call]
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))  # type: ignore[no-untyped-call]
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # type: ignore[no-untyped-call]
            ax.xaxis.set_major_locator(mdates.DayLocator())  # type: ignore[no-untyped-call]

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError:
        click.echo("Warning: matplotlib not available for PNG charts")
    except (OSError, PermissionError) as e:
        click.echo(f"Warning: Could not save PNG chart - file error: {e}")
    except ValueError as e:
        click.echo(f"Warning: Could not save PNG chart - invalid data: {e}")


def _display_monthly_results(
    detailed_df: pd.DataFrame,
    results: dict[str, Any],
    start_dt: datetime,
    end_dt: datetime,
    cost_chart: str | None = None,
    peak_kw: float | None = None,
    daily_kwh: float | None = None,
    day_kwh: float | None = None,
    night_kwh: float | None = None,
    occupancy_rate: float | None = None,
    open_file: bool = False,
    region: str | None = None,
    handling_fee: float = 0.0,
) -> None:
    """Display cost results with monthly aggregation and optional cost chart."""

    # Add month column for grouping
    detailed_df["month"] = detailed_df.index.to_period("M")

    # Calculate period length to determine output format
    period_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1

    if period_months == 1:
        # Single month - show overview
        total_kwh = results["total_consumption_kwh"]
        spot_cost = results["energy_cost"]
        handling_cost = total_kwh * handling_fee
        total_cost = spot_cost + handling_cost

        click.echo("\n=== Monthly Summary ===")
        click.echo(f"Period: {start_dt.strftime('%B %Y')}")
        click.echo(f"Consumption:      {total_kwh:>9,.0f} kWh")
        click.echo(
            f"Spot market:      €{spot_cost:>8.2f}  (€{results['weighted_avg_price_eur_kwh']:.4f}/kWh)",
        )
        if handling_fee > 0:
            click.echo(f"Handling fees:    €{handling_cost:>8.2f}  (€{handling_fee:.4f}/kWh)")
            click.echo(f"{'─' * 35}")
            click.echo(
                f"Total cost:       €{total_cost:>8.2f}  (€{total_cost / total_kwh:.4f}/kWh)",
            )

        click.echo("\n=== Price Statistics ===")
        click.echo(f"Average spot price: €{results['avg_price_eur_kwh']:.4f}/kWh")
        if handling_fee > 0:
            click.echo(f"Handling fee:       €{handling_fee:.4f}/kWh")
            click.echo(
                f"Total average:      €{results['weighted_avg_price_eur_kwh'] + handling_fee:.4f}/kWh",
            )
        click.echo(
            f"Min/Max spot:       €{results['min_price_eur_kwh']:.4f} - €{results['max_price_eur_kwh']:.4f}/kWh",
        )

    else:
        # Multi-month - show monthly breakdown
        monthly_summary = (
            detailed_df.groupby("month")
            .agg({"energy_kwh": "sum", "energy_cost_eur": "sum", "price_eur_kwh": "mean"})
            .round(4)
        )

        click.echo("\n=== Monthly Breakdown ===")
        if handling_fee > 0:
            click.echo("Month     │   kWh  │ Spot €/kWh │ Handle │ Total €/kWh │  Total €")
            click.echo("──────────┼────────┼────────────┼────────┼─────────────┼─────────")
        else:
            click.echo("Month     │   kWh  │ Spot €/kWh │  Total €")
            click.echo("──────────┼────────┼────────────┼─────────")

        for month, row in monthly_summary.iterrows():
            month_name = month.strftime("%b %Y")
            consumption = row["energy_kwh"]
            spot_cost = row["energy_cost_eur"]
            spot_price = spot_cost / consumption if consumption > 0 else 0

            if handling_fee > 0:
                handling_cost = consumption * handling_fee
                total_cost = spot_cost + handling_cost
                total_price = total_cost / consumption if consumption > 0 else 0
                click.echo(
                    f"{month_name:9} │ {consumption:6.0f} │   {spot_price:.4f}   │ {handling_fee:.4f} │   "
                    f"{total_price:.4f}   │  {total_cost:6.2f}",
                )
            else:
                click.echo(
                    f"{month_name:9} │ {consumption:6.0f} │   {spot_price:.4f}   │  {spot_cost:6.2f}",
                )

        # Overall summary
        total_kwh = results["total_consumption_kwh"]
        total_spot_cost = results["energy_cost"]
        total_handling_cost = total_kwh * handling_fee
        total_all_cost = total_spot_cost + total_handling_cost

        click.echo("\n=== Period Total ===")
        click.echo(f"Consumption:      {total_kwh:>9,.0f} kWh")
        click.echo(
            f"Spot market:      €{total_spot_cost:>8.2f}  (€{results['weighted_avg_price_eur_kwh']:.4f}/kWh)",
        )
        if handling_fee > 0:
            click.echo(f"Handling fees:    €{total_handling_cost:>8.2f}  (€{handling_fee:.4f}/kWh)")
            click.echo(f"{'─' * 35}")
            click.echo(
                f"Total cost:       €{total_all_cost:>8.2f}  (€{total_all_cost / total_kwh:.4f}/kWh)",
            )
        else:
            click.echo(f"{'─' * 35}")
            click.echo(
                f"Total cost:       €{total_spot_cost:>8.2f}  (€{results['weighted_avg_price_eur_kwh']:.4f}/kWh)",
            )

    # Generate cost chart if requested
    if cost_chart and period_months > 1:
        _save_cost_chart(
            monthly_summary,
            cost_chart,
            start_dt,
            end_dt,
            peak_kw,
            daily_kwh,
            day_kwh,
            night_kwh,
            occupancy_rate,
            open_file,
            region,
            handling_fee,
        )


def _save_cost_chart(
    monthly_summary: pd.DataFrame,
    filename: str,
    start_dt: datetime,
    end_dt: datetime,
    peak_kw: float | None = None,
    daily_kwh: float | None = None,
    day_kwh: float | None = None,
    night_kwh: float | None = None,
    occupancy_rate: float | None = None,
    open_file: bool = False,
    region: str | None = None,
    handling_fee: float = 0.0,
) -> None:
    """Save monthly cost breakdown chart as PNG."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Prepare data
        months = [month.start_time for month in monthly_summary.index]
        spot_prices = [
            cost / consumption if consumption > 0 else 0
            for cost, consumption in zip(
                monthly_summary["energy_cost_eur"],
                monthly_summary["energy_kwh"],
                strict=False,
            )
        ]
        total_prices = [spot_price + handling_fee for spot_price in spot_prices]

        # Plot monthly EUR/kWh prices
        if handling_fee > 0:
            ax.plot(
                months,
                spot_prices,
                "o-",
                linewidth=2,
                markersize=4,
                color="#1f77b4",
                alpha=0.7,
                label="Spot Market",
            )
            ax.plot(
                months,
                total_prices,
                "o-",
                linewidth=2,
                markersize=6,
                color="#d62728",
                label="Total (incl. handling)",
            )
            ax.legend()
        else:
            ax.plot(months, total_prices, "o-", linewidth=2, markersize=6, color="#1f77b4")

        # Set zero-indexed y-axis
        ax.set_ylim(bottom=0)

        # Format chart
        ax.set_title(
            f"Monthly Electricity Costs - {region.upper() if region else ''}\n"
            f"{start_dt.strftime('%b %Y')} to {end_dt.strftime('%b %Y')}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Price (EUR/kWh)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # type: ignore[no-untyped-call]
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # type: ignore[no-untyped-call]
        plt.xticks(rotation=45)

        # Add simulation parameters as text box
        params_text = []
        if peak_kw:
            params_text.append(f"Peak Power: {peak_kw:.1f} kW")
        if daily_kwh:
            params_text.append(f"Daily Consumption: {daily_kwh:.0f} kWh")
        if day_kwh and night_kwh:
            params_text.append(f"Day/Night Split: {day_kwh:.0f}/{night_kwh:.0f} kWh")
        if occupancy_rate:
            params_text.append(f"Occupancy: {occupancy_rate:.0%}")
        if handling_fee > 0:
            params_text.append(f"Handling Fee: {handling_fee:.3f} EUR/kWh")

        params_text.append("Profile: Kids Hotel")

        # Add text box with parameters
        if params_text:
            textstr = "\n".join(params_text)
            props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
            ax.text(
                0.02,
                0.98,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        click.echo(f"\nMonthly cost chart saved to: {filename}")
        if open_file:
            _open_file(filename)

    except ImportError:
        click.echo("Warning: matplotlib not available for cost charts")
    except (OSError, PermissionError) as e:
        click.echo(f"Warning: Could not save cost chart - file error: {e}")
    except ValueError as e:
        click.echo(f"Warning: Could not save cost chart - invalid data: {e}")
    except KeyError as e:
        click.echo(f"Warning: Could not save cost chart - missing data field: {e}")


@cli.command()
def sources() -> None:
    """List available data sources."""
    sources = DataSourceFactory.list_available_sources()
    click.echo("Available data sources:")
    for source in sources:
        ds = DataSourceFactory.create_data_source(source)
        info = ds.get_data_source_info()
        click.echo(f"  {source}: {info['description']}")


@cli.command()
@click.option("--region", help="Clear cache for specific region only")
@click.option(
    "--yes-i-am-a-complete-moron-who-wants-to-delete-valuable-cache-data",
    is_flag=True,
    help="Required safety flag to confirm you really want to delete cached data",
)
@click.pass_context
def clear_cache(
    ctx: click.Context,
    region: str | None,
    yes_i_am_a_complete_moron_who_wants_to_delete_valuable_cache_data: bool,
) -> None:
    """Clear cached price data."""
    if not yes_i_am_a_complete_moron_who_wants_to_delete_valuable_cache_data:
        click.echo(
            "ERROR: This command will permanently delete valuable cached data!\n"
            "If you really want to proceed, use:\n"
            "  --yes-i-am-a-complete-moron-who-wants-to-delete-valuable-cache-data\n"
            "\nThis safety flag exists because some idiot once accidentally deleted\n"
            "265 cache files while testing code changes. Don't be that idiot.",
            err=True,
        )
        ctx.exit(1)

    data_source = DataSourceFactory.create_data_source(ctx.obj["data_source"], ctx.obj["cache_dir"])

    # Show what will be deleted first
    from .data.cache import DataCacheManager

    cache_manager = DataCacheManager(ctx.obj["cache_dir"] or "data/cache")
    cache_info = cache_manager.get_cache_info()

    click.echo("\n⚠️  ABOUT TO DELETE:")
    click.echo(f"   📁 {cache_info['total_files']} cache files")
    click.echo(f"   💾 {cache_info['total_size_mb']:.1f} MB of data")
    if region:
        click.echo(f"   🌍 Region: {region} only")
    else:
        click.echo("   🌍 ALL regions and data types")
    types_str = ", ".join(f"{src}_{typ}" for src, types in cache_info["by_source_type"].items() for typ in types)
    click.echo(f"   🗂️  Types: {types_str}")

    if not click.confirm("\nAre you ABSOLUTELY SURE you want to permanently delete this cached data?", default=False):
        click.echo("❌ Cancelled. Cache data preserved.")
        ctx.exit(0)

    data_source.clear_cache(region)

    if region:
        click.echo(f"✅ Cleared cache for region: {region}")
    else:
        click.echo("✅ Cleared all cached data")


@cli.command()
@add_standard_options
@add_duck_factor_options
@click.option("--open", "open_file", is_flag=True, help="Open generated PNG file after creation")
@click.pass_context
def duck_factor(
    ctx: click.Context,
    region: str,
    start_date: str,
    end_date: str | None,
    window: str,
    step: str,
    output: str | None,
    png: bool,
    chart_type: str,
    no_cache: bool,
    open_file: bool,
    width: int | None,
    height: int | None,
) -> None:
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
            ctx.obj["data_source"],
            ctx.obj["cache_dir"],
        )

        click.echo(f"Fetching price data for {region} from {date_description}...")
        click.echo(f"Rolling window: {window_days} days, step: {step_days} days")

        # Get price data
        df = data_source.get_spot_prices(region, start_dt, end_dt, use_cache=not no_cache)

        click.echo(f"Retrieved {len(df)} hourly price points")

        # Perform rolling duck factor analysis
        click.echo("Analyzing duck curve evolution...")
        analysis_results = analyze_rolling_duck_patterns(df, region, window_days, step_days)

        if "error" in analysis_results:
            click.echo(f"Analysis error: {analysis_results['error']}", err=True)
            ctx.exit(1)

        duck_factors = analysis_results["duck_factors"]
        seasonal_data = analysis_results["seasonal_patterns"]
        trends = analysis_results["trends"]
        yoy_data = analysis_results["year_over_year"]

        click.echo(f"Generated {len(duck_factors)} duck factor data points")

        # Display analysis summary
        _display_duck_factor_summary(duck_factors, trends, seasonal_data, yoy_data)

        # Generate charts
        if png or output:
            # PNG output mode - use smart filename generation
            try:
                if chart_type == "time-series":
                    if png:
                        output = generate_duck_factor_filename(
                            region,
                            start_date_str,
                            end_date_str,
                            window_days,
                            "timeseries",
                        )
                    elif output is not None and not output.lower().endswith(".png"):
                        output += ".png"
                    if output is not None:
                        create_png_duck_factor_chart(
                            duck_factors,
                            region,
                            output,
                            window_days,
                            width=width or 12,
                            height=height or 6,
                        )
                        if open_file:
                            _open_file(output)
                elif chart_type == "seasonal":
                    if png:
                        output = generate_duck_factor_filename(
                            region,
                            start_date_str,
                            end_date_str,
                            window_days,
                            "seasonal",
                        )
                    elif output is not None and not output.lower().endswith(".png"):
                        output += ".png"
                    if output is not None:
                        create_png_seasonal_duck_chart(
                            seasonal_data,
                            region,
                            output,
                            width=width or 12,
                            height=height or 8,
                        )
                        if open_file:
                            _open_file(output)
                elif chart_type == "multi-window":
                    # Create multiple window analysis with smart filenames
                    from .analysis.rolling_duck import RollingDuckAnalyzer

                    analyzer = RollingDuckAnalyzer(region)
                    windows = [7, 30, 90]
                    multi_results = analyzer.multi_window_analysis(df, windows, step_days)

                    window_filenames = get_multi_window_filenames(
                        region,
                        start_date_str,
                        end_date_str,
                        [f"{w}d" for w in windows],
                    )

                    for window_key, window_df in multi_results.items():
                        window_output = window_filenames[window_key]
                        create_png_duck_factor_chart(
                            window_df,
                            region,
                            window_output,
                            int(window_key[:-1]),
                            width=width or 12,
                            height=height or 6,
                        )
                        if open_file:
                            _open_file(window_output)
                elif chart_type == "all":
                    # Create all chart types with smart filenames
                    timeseries_output = generate_duck_factor_filename(
                        region,
                        start_date_str,
                        end_date_str,
                        window_days,
                        "timeseries",
                    )
                    seasonal_output = generate_duck_factor_filename(
                        region,
                        start_date_str,
                        end_date_str,
                        window_days,
                        "seasonal",
                    )

                    create_png_duck_factor_chart(
                        duck_factors,
                        region,
                        timeseries_output,
                        window_days,
                        width=width or 12,
                        height=height or 6,
                    )
                    create_png_seasonal_duck_chart(
                        seasonal_data,
                        region,
                        seasonal_output,
                        width=width or 12,
                        height=height or 8,
                    )

                    if open_file:
                        _open_file(timeseries_output)
                        _open_file(seasonal_output)

            except ImportError as e:
                click.echo(f"Error: {e}", err=True)
                click.echo("Please install matplotlib: pip install matplotlib>=3.7.0")
                ctx.exit(1)
        else:
            # Terminal output mode (default)
            if chart_type == "time-series" or chart_type == "all":
                create_terminal_duck_factor_chart(
                    duck_factors,
                    region,
                    window_days,
                    width=width,
                    height=height,
                )

            # Additional terminal analysis summaries for other chart types
            if chart_type == "seasonal" or chart_type == "all":
                _display_seasonal_analysis(seasonal_data, region)

            if chart_type == "multi-window" or chart_type == "all":
                _display_multi_window_analysis(df, region, step_days)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except OSError as e:
        click.echo(f"File I/O error: {e}", err=True)
        ctx.exit(1)
    except (ConnectionError, TimeoutError) as e:
        click.echo(f"Network error while fetching data: {e}", err=True)
        ctx.exit(1)
    except pd.errors.EmptyDataError as e:
        click.echo(f"No data available for the specified period: {e}", err=True)
        ctx.exit(1)
    except KeyError as e:
        click.echo(f"Data format error - missing expected field: {e}", err=True)
        ctx.exit(1)


def _parse_time_period(period_str: str) -> int:
    """Parse time period string like '30d', '7d' into number of days."""
    period_str = period_str.lower().strip()

    if period_str.endswith("d"):
        return int(period_str[:-1])
    if period_str.endswith("w"):
        return int(period_str[:-1]) * 7
    if period_str.endswith("m"):
        return int(period_str[:-1]) * 30  # Approximate
    if period_str.endswith("y"):
        return int(period_str[:-1]) * 365  # Approximate
    # Assume it's just a number of days
    return int(period_str)


def _display_duck_factor_summary(
    duck_factors: pd.DataFrame,
    trends: dict[str, Any],
    seasonal_data: dict[str, Any],
    yoy_data: dict[str, Any],
) -> None:
    """Display a summary of duck factor analysis results."""
    click.echo("\n" + "=" * 60)
    click.echo("DUCK FACTOR ANALYSIS SUMMARY")
    click.echo("=" * 60)

    if not duck_factors.empty:
        avg_factor = duck_factors["duck_factor"].mean()
        min_factor = duck_factors["duck_factor"].min()
        max_factor = duck_factors["duck_factor"].max()

        click.echo(f"Average Duck Factor: {avg_factor:.3f}")
        click.echo(f"Range: {min_factor:.3f} - {max_factor:.3f}")

        # Trend information
        if "trend_slope_per_year" in trends:
            slope = trends["trend_slope_per_year"]
            r_squared = trends.get("trend_r_squared", 0)
            classification = trends.get("trend_classification", "Unknown")

            click.echo(f"Trend: {classification} ({slope:+.4f}/year, R²={r_squared:.3f})")

        # Seasonal information
        if seasonal_data and "peak_season" in seasonal_data:
            peak_season = seasonal_data["peak_season"]
            low_season = seasonal_data["low_season"]
            seasonal_range = seasonal_data.get("seasonal_range", 0)

            click.echo(f"Seasonal Pattern: Peak in {peak_season}, Low in {low_season}")
            click.echo(f"Seasonal Range: {seasonal_range:.3f}")

        # Year-over-year
        if yoy_data and "avg_annual_change" in yoy_data:
            avg_change = yoy_data["avg_annual_change"]
            click.echo(f"Average Annual Change: {avg_change:+.1%}")

    click.echo("=" * 60)


def _display_seasonal_analysis(seasonal_data: dict[str, Any], region: str) -> None:
    """Display seasonal analysis in terminal format."""
    if not seasonal_data or seasonal_data.get("seasonal_patterns", pd.DataFrame()).empty:
        click.echo("No seasonal data available")
        return

    click.echo(f"\n{'=' * 50}")
    click.echo(f"SEASONAL DUCK FACTOR ANALYSIS - {_get_country_name(region)}")
    click.echo(f"{'=' * 50}")

    seasonal_df = seasonal_data["seasonal_patterns"]
    for _, row in seasonal_df.iterrows():
        click.echo(
            f"{row['season']:<8}: {row['mean']:.3f} ±{row['std']:.3f} ({row['count']} samples)",
        )

    if "peak_season" in seasonal_data:
        click.echo(f"\nPeak Season: {seasonal_data['peak_season']}")
        click.echo(f"Low Season: {seasonal_data['low_season']}")
        click.echo(f"Seasonal Range: {seasonal_data.get('seasonal_range', 0):.3f}")


def _display_multi_window_analysis(df: pd.DataFrame, region: str, step_days: int) -> None:
    """Display multi-window analysis summary."""
    from .analysis.rolling_duck import RollingDuckAnalyzer

    analyzer = RollingDuckAnalyzer(region)
    multi_results = analyzer.multi_window_analysis(df, [7, 30, 90], step_days)

    click.echo(f"\n{'=' * 50}")
    click.echo("MULTI-WINDOW DUCK FACTOR COMPARISON")
    click.echo(f"{'=' * 50}")

    for window_key, window_df in multi_results.items():
        if not window_df.empty:
            avg_factor = window_df["duck_factor"].mean()
            volatility = window_df["duck_factor"].std()
            click.echo(f"{window_key:<8}: avg={avg_factor:.3f}, volatility={volatility:.3f}")


@cli.command()
@add_standard_options
@add_negative_pricing_options
@click.option("--open", "open_file", is_flag=True, help="Open generated PNG file after creation")
@click.pass_context
def negative_pricing(
    ctx: click.Context,
    region: str,
    start_date: str,
    end_date: str | None,
    threshold: float,
    severe_threshold: float,
    extreme_threshold: float,
    cheap_threshold: float,
    chart_type: str,
    aggregation_level: str,
    output: str | None,
    png: bool,
    no_cache: bool,
    width: int | None,
    height: int | None,
    open_file: bool,
) -> None:
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
            ctx.obj["data_source"],
            ctx.obj["cache_dir"],
        )

        click.echo(f"Fetching price data for {region} from {date_description}...")
        click.echo(f"Near-zero threshold: {threshold} EUR/MWh")

        # Get price data
        df = data_source.get_spot_prices(region, start_dt, end_dt, use_cache=not no_cache)

        click.echo(f"Retrieved {len(df)} hourly price points")

        # Handle different chart types
        if chart_type == "timechart":
            # Timechart mode - no need for comprehensive analysis
            click.echo(f"Generating {aggregation_level} hours timechart...")

            # Handle PNG output - either via --png flag or --output
            if png or output:
                # PNG output mode
                if png:
                    # Auto-generate filename when using --png flag
                    output = (
                        f"images/negative-pricing-timechart-{aggregation_level}_{region.lower()}_"
                        f"{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}.png"
                    )
                elif output is not None and not output.lower().endswith(".png"):
                    # Add .png extension if not present
                    output += ".png"

                try:
                    if output is not None:
                        create_png_negative_pricing_timechart(
                            df,
                            region,
                            output,
                            width=width or 12,
                            height=height or 6,
                            near_zero_threshold=threshold,
                            severe_threshold=severe_threshold,
                            extreme_threshold=extreme_threshold,
                            cheap_threshold=cheap_threshold,
                            aggregation_level=aggregation_level,
                        )
                        if open_file:
                            _open_file(output)
                except ImportError as e:
                    click.echo(f"Error: {e}", err=True)
                    click.echo("Please install matplotlib: pip install matplotlib>=3.7.0")
                    ctx.exit(1)
            else:
                # Terminal output mode (default)
                create_terminal_negative_pricing_timechart(
                    df,
                    region,
                    width=width,
                    height=height,
                    near_zero_threshold=threshold,
                    severe_threshold=severe_threshold,
                    extreme_threshold=extreme_threshold,
                    cheap_threshold=cheap_threshold,
                    aggregation_level=aggregation_level,
                )
        else:
            # Analysis mode (default)
            click.echo("Analyzing negative pricing patterns...")
            analysis_results = analyze_negative_pricing_comprehensive(df, region, threshold)

            if "error" in analysis_results:
                click.echo(f"Analysis error: {analysis_results['error']}", err=True)
                ctx.exit(1)

            metrics = analysis_results["overall_metrics"]
            seasonal_data = analysis_results["seasonal_patterns"]

            # Display analysis summary
            _display_negative_pricing_summary(metrics, seasonal_data, region)

            # Generate charts
            if png or output:
                # PNG output mode
                if png:
                    # Auto-generate filename when using --png flag
                    output = (
                        f"images/negative-pricing_{region.lower()}_"
                        f"{start_date_str.replace('-', '')}_{end_date_str.replace('-', '')}.png"
                    )
                elif output is not None and not output.lower().endswith(".png"):
                    # Add .png extension if not present
                    output += ".png"

                try:
                    if output is not None:
                        create_png_negative_pricing_chart(
                            df,
                            region,
                            output,
                            width=width or 12,
                            height=height or 8,
                            near_zero_threshold=threshold,
                        )
                        if open_file:
                            _open_file(output)
                except ImportError as e:
                    click.echo(f"Error: {e}", err=True)
                    click.echo("Please install matplotlib: pip install matplotlib>=3.7.0")
                    ctx.exit(1)
            else:
                # Terminal output mode (default)
                create_terminal_negative_pricing_chart(
                    df,
                    region,
                    width=width,
                    height=height,
                    near_zero_threshold=threshold,
                )

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except OSError as e:
        click.echo(f"File I/O error: {e}", err=True)
        ctx.exit(1)
    except (ConnectionError, TimeoutError) as e:
        click.echo(f"Network error while fetching data: {e}", err=True)
        ctx.exit(1)
    except pd.errors.EmptyDataError as e:
        click.echo(f"No data available for the specified period: {e}", err=True)
        ctx.exit(1)
    except KeyError as e:
        click.echo(f"Data format error - missing expected field: {e}", err=True)
        ctx.exit(1)


def _display_negative_pricing_summary(metrics: Any, seasonal_data: dict[str, Any], region: str) -> None:
    """Display comprehensive negative pricing analysis summary."""
    click.echo("\n" + "=" * 70)
    click.echo("NEGATIVE PRICING ANALYSIS SUMMARY")
    click.echo("=" * 70)

    # Overall statistics
    click.echo(
        f"Total negative hours: {metrics.negative_hours} ({metrics.negative_percentage:.1f}%)",
    )
    click.echo(
        f"Total near-zero hours: {metrics.near_zero_hours} ({metrics.near_zero_percentage:.1f}%)",
    )
    click.echo(f"Average negative hours per day: {metrics.avg_hours_per_day:.1f}")
    click.echo(f"Maximum consecutive negative hours: {metrics.max_consecutive_hours}")

    # Peak negative pricing hours
    if metrics.hourly_breakdown:
        peak_hour = max(
            metrics.hourly_breakdown.keys(),
            key=lambda h: metrics.hourly_breakdown[h].get("negative_percentage", 0),
        )
        peak_percentage = metrics.hourly_breakdown[peak_hour]["negative_percentage"]
        click.echo(
            f"Peak negative pricing hour: {peak_hour}:00 ({peak_percentage:.1f}% of the time)",
        )

    # Monthly insights
    if metrics.monthly_breakdown:
        monthly_avg = [(month, data["avg_hours_per_day"]) for month, data in metrics.monthly_breakdown.items()]
        monthly_avg.sort(key=lambda x: x[1], reverse=True)

        best_month = monthly_avg[0]
        worst_month = monthly_avg[-1]

        month_names = [
            "",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        click.echo(
            f"Best month for negative pricing: {month_names[best_month[0]]} ({best_month[1]:.1f} hours/day)",
        )
        click.echo(
            f"Worst month for negative pricing: {month_names[worst_month[0]]} ({worst_month[1]:.1f} hours/day)",
        )

    # Solar potential analysis
    if "yearly_summary" in seasonal_data:
        yearly = seasonal_data["yearly_summary"]
        click.echo("\nSOLAR POTENTIAL ANALYSIS (based on approximate solar irradiation data):")
        click.echo(f"Current average: {yearly['avg_current_hours_per_day']:.1f} hours/day")
        click.echo(
            f"Theoretical maximum: {yearly['avg_theoretical_max_hours_per_day']:.1f} hours/day",
        )
        click.echo(f"Overall progress: {yearly['overall_progress_percentage']:.1f}%")

        remaining = yearly["avg_theoretical_max_hours_per_day"] - yearly["avg_current_hours_per_day"]
        click.echo(f"Remaining potential: {remaining:.1f} hours/day")
        click.echo(
            "NOTE: Solar potential estimates based on EU PVGIS, Global Solar Atlas, and Copernicus data",
        )

    # Regional insights
    click.echo(f"\nREGIONAL CONTEXT ({_get_country_name(region)}):")

    # Show seasonal variation
    if len(seasonal_data) > 1:  # More than just yearly_summary
        summer_months = [6, 7, 8]  # June, July, August
        winter_months = [12, 1, 2]  # Dec, Jan, Feb

        summer_values = [
            seasonal_data[str(m)]["progress_metrics"]["current_hours_per_day"]
            for m in summer_months
            if str(m) in seasonal_data and "error" not in seasonal_data[str(m)]["progress_metrics"]
        ]
        winter_values = [
            seasonal_data[str(m)]["progress_metrics"]["current_hours_per_day"]
            for m in winter_months
            if str(m) in seasonal_data and "error" not in seasonal_data[str(m)]["progress_metrics"]
        ]

        summer_avg = np.mean(summer_values) if summer_values else np.nan
        winter_avg = np.mean(winter_values) if winter_values else np.nan

        if not np.isnan(summer_avg) and not np.isnan(winter_avg):
            seasonal_ratio = summer_avg / winter_avg if winter_avg > 0 else float("inf")
            click.echo(
                f"Summer vs Winter ratio: {seasonal_ratio:.1f}x (Summer: {summer_avg:.1f}h, Winter: {winter_avg:.1f}h)",
            )

    click.echo("=" * 70)


@cli.command()
@click.option("--region", required=True, help="Region code (e.g., DE, AT, FR)")
@click.option("--start-year", type=int, required=True, help="Starting year for analysis")
@click.option("--end-year", type=int, required=True, help="Ending year for analysis (inclusive)")
@click.option("--reference-year", type=int, help="Reference year for peak migration analysis")
@click.option("--percentile", type=float, default=95.0, help="Percentile for peak calculation (default: 95)")
@click.option("--output", help="Base filename for saving charts (auto-generates extensions)")
@click.option("--no-cache", is_flag=True, help="Skip cache and fetch fresh data")
@click.option(
    "--chart-type",
    type=click.Choice(["hourly-evolution", "duck-curve", "peak-migration", "all"]),
    default="hourly-evolution",
    help="Type of chart to generate",
)
@click.option("--open", "open_file", is_flag=True, help="Open generated PNG file after creation")
@click.pass_context
def load_peaks(
    ctx: click.Context,
    region: str,
    start_year: int,
    end_year: int,
    reference_year: int | None,
    percentile: float,
    output: str | None,
    no_cache: bool,  # noqa: ARG001
    chart_type: str,
    open_file: bool,
) -> None:
    """Analyze hourly peak load evolution to detect behind-the-meter PV impacts."""
    try:
        from .analysis.load_patterns import LoadPatternAnalyzer
        from .charts.load_patterns import LoadPatternCharts

        # Initialize analyzer
        analyzer: LoadPatternAnalyzer = LoadPatternAnalyzer()

        click.echo(f"Fetching load data for {region} from {start_year}-{end_year}...")

        # Fetch multi-year data
        load_data, coverage_info = analyzer.fetch_multi_year_load_data(region, start_year, end_year)

        if load_data.empty:
            click.echo("No load data available for the specified period")
            ctx.exit(1)

        # Display coverage information
        start_str = coverage_info["actual_start_date"].strftime("%Y-%m-%d")
        end_str = coverage_info["actual_end_date"].strftime("%Y-%m-%d")

        if coverage_info["missing_years"]:
            click.echo(
                f"📊 Data Coverage: {start_str} to {end_str} ({len(coverage_info['missing_years'])} years missing)",
            )
            click.echo(f"   Missing years: {coverage_info['missing_years']}")
        else:
            click.echo(f"📊 Data Coverage: {start_str} to {end_str} (complete)")

        click.echo(f"Analyzing {len(load_data)} hours of load data...")

        # Calculate hourly peaks
        hourly_peaks = analyzer.calculate_hourly_peaks(load_data, group_by="year", percentile=percentile)

        # Calculate duck curve evolution
        duck_metrics = analyzer.detect_duck_curve_formation(load_data)

        # Analyze peak migration if reference year specified
        migration_analysis = None
        if reference_year:
            try:
                migration_analysis = analyzer.analyze_peak_migration(hourly_peaks, reference_year=reference_year)
            except ValueError as e:
                click.echo(f"Peak migration analysis error: {e}")

        # Display summary
        _display_load_analysis_summary(hourly_peaks, duck_metrics, migration_analysis, region, coverage_info)

        # Generate charts
        if chart_type == "hourly-evolution" or chart_type == "all":
            if output == "auto":
                filter_str = f"p{int(percentile)}" if percentile != 100 else None
                output_file = generate_load_peaks_filename(region, coverage_info, "hourly-evolution", filter_str)
            elif output:
                output_file = f"{output}_hourly_evolution.png"
            else:
                output_file = None
            LoadPatternCharts.plot_hourly_peaks_evolution(
                hourly_peaks,
                region,
                coverage_info,
                percentile=percentile,
                output_file=output_file,
            )
            if open_file and output_file:
                _open_file(output_file)

        if chart_type == "duck-curve" or chart_type == "all":
            if output == "auto":
                filter_str = f"p{int(percentile)}" if percentile != 100 else None
                output_file = generate_load_peaks_filename(region, coverage_info, "duck-curve", filter_str)
            elif output:
                output_file = f"{output}_duck_curve.png"
            else:
                output_file = None
            LoadPatternCharts.plot_duck_curve_evolution(duck_metrics, region, output_file=output_file)
            if open_file and output_file:
                _open_file(output_file)

        if (chart_type == "peak-migration" or chart_type == "all") and not hourly_peaks.empty:
            ref_year = reference_year or hourly_peaks["year"].min()
            if output == "auto":
                filter_str = f"p{int(percentile)}" if percentile != 100 else None
                output_file = generate_load_peaks_filename(region, coverage_info, "peak-migration", filter_str)
            elif output:
                output_file = f"{output}_peak_migration.png"
            else:
                output_file = None
            LoadPatternCharts.plot_peak_migration_heatmap(
                hourly_peaks,
                region,
                reference_year=ref_year,
                output_file=output_file,
            )
            if open_file and output_file:
                _open_file(output_file)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    except (ConnectionError, TimeoutError) as e:
        click.echo(f"Network error while fetching data: {e}", err=True)
        ctx.exit(1)
    except ImportError as e:
        click.echo(f"Missing dependency: {e}", err=True)
        ctx.exit(1)


def _display_load_analysis_summary(
    hourly_peaks: pd.DataFrame,
    duck_metrics: pd.DataFrame,
    migration_analysis: pd.DataFrame | None,
    region: str,
    coverage_info: dict[str, Any],
) -> None:
    """Display load pattern analysis summary."""
    click.echo("\n" + "=" * 70)
    click.echo("LOAD PATTERN ANALYSIS SUMMARY")
    click.echo("=" * 70)

    # Show actual coverage vs requested
    start_str = coverage_info["actual_start_date"].strftime("%Y-%m-%d")
    end_str = coverage_info["actual_end_date"].strftime("%Y-%m-%d")
    actual_range = f"{start_str} to {end_str}"
    click.echo(f"Analysis period: {actual_range} ({region})")
    if coverage_info["missing_years"]:
        click.echo(f"Missing data: {len(coverage_info['missing_years'])} years ({coverage_info['missing_years']})")

    if not hourly_peaks.empty:
        years = sorted(hourly_peaks["year"].unique())
        click.echo(f"Data points: {len(hourly_peaks)} hourly peak measurements")

        # Peak load trends
        first_year_peaks = hourly_peaks[hourly_peaks["year"] == years[0]]
        last_year_peaks = hourly_peaks[hourly_peaks["year"] == years[-1]]

        if not first_year_peaks.empty and not last_year_peaks.empty:
            first_max = first_year_peaks["peak_load"].max() / 1000  # Convert to GW
            last_max = last_year_peaks["peak_load"].max() / 1000
            peak_change = ((last_max - first_max) / first_max * 100) if first_max > 0 else 0

            peak_msg = (
                f"Peak load change: {first_max:.1f} GW ({years[0]}) → "
                f"{last_max:.1f} GW ({years[-1]}) ({peak_change:+.1f}%)"
            )
            click.echo(peak_msg)

    if not duck_metrics.empty:
        click.echo("\nDUCK CURVE EVOLUTION:")
        first_year = duck_metrics.iloc[0]
        last_year = duck_metrics.iloc[-1]

        click.echo(f"Duck intensity: {first_year['duck_intensity_pct']:.1f}% → {last_year['duck_intensity_pct']:.1f}%")
        suppression_msg = (
            f"Midday suppression: {first_year['midday_suppression_pct']:.1f}% → "
            f"{last_year['midday_suppression_pct']:.1f}%"
        )
        click.echo(suppression_msg)

        # Calculate trends
        if len(duck_metrics) > 1:
            years = duck_metrics["year"]
            intensity_trend = np.polyfit(years, duck_metrics["duck_intensity_pct"], 1)[0]
            click.echo(f"Duck curve trend: {intensity_trend:+.2f}% intensity per year")

    if migration_analysis is not None and not migration_analysis.empty:
        click.echo("\nPEAK MIGRATION PATTERNS:")
        recent_year = migration_analysis.iloc[-1]

        morning_change = recent_year["morning_peak_change"]
        evening_change = recent_year["evening_peak_change"]
        midday_change = recent_year["midday_change"]

        click.echo(f"Morning peaks (6-9 AM): {morning_change:+.1f}% vs reference")
        click.echo(f"Evening peaks (5-8 PM): {evening_change:+.1f}% vs reference")
        click.echo(f"Midday demand (11 AM-2 PM): {midday_change:+.1f}% vs reference")

        if midday_change < -5 and (morning_change > 0 or evening_change > 0):
            click.echo("🔍 Pattern suggests behind-the-meter PV impact: midday suppression with peak preservation")

    click.echo("=" * 70)


def _get_country_name(region_code: str) -> str:
    """Get country name from region code."""
    country_names = {
        "AT": "Austria",
        "BE": "Belgium",
        "CH": "Switzerland",
        "CZ": "Czech Republic",
        "DE": "Germany",
        "DK": "Denmark",
        "ES": "Spain",
        "FR": "France",
        "IT": "Italy",
        "NL": "Netherlands",
        "NO": "Norway",
        "PL": "Poland",
        "SE": "Sweden",
        "UK": "United Kingdom",
    }
    return country_names.get(region_code.upper(), region_code)


if __name__ == "__main__":
    cli()
