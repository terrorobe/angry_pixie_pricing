"""Terminal-based charting using plotext."""

import plotext as plt
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from ..analysis.hourly import HourlyPriceAnalyzer
from ..analysis.rolling_duck import RollingDuckAnalyzer
from ..analysis.negative_pricing import NegativePricingAnalyzer

# Try to import matplotlib for PNG export, fall back gracefully
try:
    import matplotlib.pyplot as mpl_plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def _get_country_name(region_code: str) -> str:
    """
    Convert region code to full country name.
    
    Args:
        region_code: Two-letter region/country code
        
    Returns:
        Full country name or original code if not found
    """
    country_names = {
        'AT': 'Austria',
        'BE': 'Belgium', 
        'CH': 'Switzerland',
        'CZ': 'Czech Republic',
        'DE': 'Germany',
        'DK': 'Denmark',
        'ES': 'Spain',
        'FR': 'France',
        'IT': 'Italy',
        'NL': 'Netherlands',
        'NO': 'Norway',
        'PL': 'Poland',
        'SE': 'Sweden',
        'UK': 'United Kingdom'
    }
    return country_names.get(region_code.upper(), region_code)


def create_png_price_chart(
    df: pd.DataFrame,
    region: str,
    output_path: str,
    title: Optional[str] = None,
    width: int = 12,
    height: int = 6,
) -> None:
    """
    Create a PNG price chart using matplotlib.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for the chart title
        output_path: Path to save the PNG file
        title: Optional custom title
        width: Figure width in inches
        height: Figure height in inches
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for PNG chart generation")
    
    if df.empty:
        print(f"No data available for {region}")
        return
    
    # Create figure and axis
    fig, ax = mpl_plt.subplots(figsize=(width, height))
    
    # Plot the data
    ax.plot(df['timestamp'], df['price'], marker='o', markersize=2, linewidth=1.5, color='#1f77b4')
    
    # Set title
    if title is None:
        start_date = df["timestamp"].min().strftime("%Y-%m-%d")
        end_date = df["timestamp"].max().strftime("%Y-%m-%d")
        country_name = _get_country_name(region)
        title = f"Electricity Spot Prices - {country_name} ({start_date} to {end_date})"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(f"Price ({df['unit'].iloc[0] if not df.empty else 'EUR/MWh'})", fontsize=12)
    
    # Format x-axis for better date display
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(df) // 10)))
    mpl_plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    mpl_plt.tight_layout()
    mpl_plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mpl_plt.close(fig)
    
    print(f"Chart saved to: {output_path}")


def create_png_hourly_analysis_chart(
    df: pd.DataFrame,
    region: str,
    output_path: str,
    width: int = 12,
    height: int = 6,
) -> None:
    """
    Create PNG duck curve analysis chart using matplotlib.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for holiday detection and display
        output_path: Path to save the PNG file
        width: Figure width in inches
        height: Figure height in inches
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for PNG chart generation")
        
    if df.empty:
        print(f"No data available for {region}")
        return
    
    analyzer = HourlyPriceAnalyzer(region)
    results = analyzer.analyze_hourly_patterns(df)
    
    if not results or 'workday' not in results or 'non_workday' not in results:
        print("Insufficient data for duck curve analysis")
        return
    
    # Create figure and axis
    fig, ax = mpl_plt.subplots(figsize=(width, height))
    
    workday_stats = results['workday']
    nonworkday_stats = results['non_workday']
    unit = df['unit'].iloc[0] if not df.empty else 'EUR/MWh'
    
    hours = list(range(24))
    workday_prices = [workday_stats[workday_stats['hour'] == h]['mean'].iloc[0] if not workday_stats[workday_stats['hour'] == h].empty else 0 for h in hours]
    nonworkday_prices = [nonworkday_stats[nonworkday_stats['hour'] == h]['mean'].iloc[0] if not nonworkday_stats[nonworkday_stats['hour'] == h].empty else 0 for h in hours]
    
    # Plot both lines
    ax.plot(hours, workday_prices, marker='o', markersize=4, linewidth=2, label='Workdays', color='#d62728')
    ax.plot(hours, nonworkday_prices, marker='s', markersize=4, linewidth=2, label='Weekends/Holidays', color='#2ca02c')
    
    # Set title and labels
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    country_name = _get_country_name(region)
    ax.set_title(f"Duck Curve Analysis - {country_name} ({start_date} to {end_date})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel(f"Average Price ({unit})", fontsize=12)
    
    # Set x-axis ticks
    ax.set_xticks(range(0, 24, 4))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 4)])
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Adjust layout and save
    mpl_plt.tight_layout()
    mpl_plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mpl_plt.close(fig)
    
    print(f"Duck curve chart saved to: {output_path}")


def create_png_hourly_workday_chart(
    df: pd.DataFrame,
    region: str,
    output_path: str,
    width: int = 12,
    height: int = 6,
) -> None:
    """
    Create PNG workday-only duck curve chart using matplotlib.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for holiday detection and display
        output_path: Path to save the PNG file
        width: Figure width in inches
        height: Figure height in inches
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for PNG chart generation")
        
    if df.empty:
        print(f"No data available for {region}")
        return
    
    analyzer = HourlyPriceAnalyzer(region)
    results = analyzer.analyze_hourly_patterns(df)
    
    if 'workday' not in results:
        print("No workday data available for analysis")
        return
    
    workday_stats = results['workday']
    unit = df['unit'].iloc[0] if not df.empty else 'EUR/MWh'
    
    # Create figure and axis
    fig, ax = mpl_plt.subplots(figsize=(width, height))
    
    hours = list(range(24))
    prices = [workday_stats[workday_stats['hour'] == h]['mean'].iloc[0] if not workday_stats[workday_stats['hour'] == h].empty else 0 for h in hours]
    
    # Plot the line
    ax.plot(hours, prices, marker='o', markersize=4, linewidth=2, color='#1f77b4')
    
    # Set title and labels
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    country_name = _get_country_name(region)
    ax.set_title(f"Workday Duck Curve - {country_name} ({start_date} to {end_date})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel(f"Average Price ({unit})", fontsize=12)
    
    # Set x-axis ticks
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 2)])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    mpl_plt.tight_layout()
    mpl_plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mpl_plt.close(fig)
    
    print(f"Workday duck curve chart saved to: {output_path}")


def create_terminal_price_chart(
    df: pd.DataFrame,
    region: str,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Create a terminal-based price chart using plotext.

    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for the chart title
        title: Optional custom title
        width: Chart width (defaults to terminal width)
        height: Chart height (defaults to auto)
    """
    # Clear any previous plots
    plt.clear_data()
    plt.clear_figure()

    # Convert timestamps to string labels for x-axis
    timestamps = df["timestamp"].dt.strftime("%m-%d %H:%M")
    prices = df["price"].tolist()
    indices = list(range(len(prices)))

    # Create the plot - connect discrete points with straight lines, no curve interpolation
    plt.plot(indices, prices, marker="hd", color="cyan")

    # Set title
    if title is None:
        start_date = df["timestamp"].min().strftime("%Y-%m-%d")
        end_date = df["timestamp"].max().strftime("%Y-%m-%d")
        unit = df["unit"].iloc[0] if not df.empty else "EUR/MWh"
        country_name = _get_country_name(region)
        title = f"Electricity Spot Prices - {country_name} ({start_date} to {end_date})"

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(f"Price ({df['unit'].iloc[0] if not df.empty else 'EUR/MWh'})")

    # Set chart size if specified
    if width:
        plt.plotsize(width, height or 20)
    elif height:
        plt.plotsize(None, height)

    # Configure x-axis to show fewer labels for readability
    total_points = len(timestamps)
    if total_points > 20:
        # Show every nth label to avoid crowding
        step = max(1, total_points // 10)
        x_indices = [i for i in range(0, total_points, step)]
        x_labels = [timestamps[i] for i in x_indices]
        plt.xticks(x_indices, x_labels)
    else:
        plt.xticks(indices, timestamps)

    # Add grid for better readability
    plt.grid(True, True)

    # Show the plot
    plt.show()


def create_terminal_price_summary(df: pd.DataFrame, region: str) -> None:
    """
    Create a terminal-based summary table of price statistics.

    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for display
    """
    if df.empty:
        print(f"No data available for {region}")
        return

    unit = df["unit"].iloc[0]
    stats = df["price"].describe()

    print(f"\n📊 Price Statistics for {region}")
    print("=" * 40)
    print(
        f"Period: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"Data points: {len(df)}")
    print()
    print(f"Mean:     {stats['mean']:.2f} {unit}")
    print(f"Median:   {stats['50%']:.2f} {unit}")
    print(f"Min:      {stats['min']:.2f} {unit}")
    print(f"Max:      {stats['max']:.2f} {unit}")
    print(f"Std Dev:  {stats['std']:.2f} {unit}")
    print()

    # Find negative price periods
    negative_prices = df[df["price"] < 0]
    if not negative_prices.empty:
        print(
            f"⚠️  Negative prices: {len(negative_prices)} hours ({len(negative_prices)/len(df)*100:.1f}%)"
        )
        print(f"   Lowest: {negative_prices['price'].min():.2f} {unit}")
        print()

    # Find peak hours
    peak_price = df.loc[df["price"].idxmax()]
    low_price = df.loc[df["price"].idxmin()]

    print(
        f"🔴 Highest price: {peak_price['price']:.2f} {unit} at {peak_price['timestamp'].strftime('%Y-%m-%d %H:%M')}"
    )
    print(
        f"🟢 Lowest price:  {low_price['price']:.2f} {unit} at {low_price['timestamp'].strftime('%Y-%m-%d %H:%M')}"
    )
    print()


def create_terminal_daily_average_chart(df: pd.DataFrame, region: str) -> None:
    """
    Create a terminal chart showing daily average prices.

    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for display
    """
    if df.empty:
        print(f"No data available for {region}")
        return

    # Calculate daily averages
    df_copy = df.copy()
    df_copy["date"] = df_copy["timestamp"].dt.date
    daily_avg = df_copy.groupby("date")["price"].mean().reset_index()

    if len(daily_avg) < 2:
        print("Need at least 2 days of data for daily averages")
        return

    plt.clear_data()
    plt.clear_figure()

    dates = [str(d) for d in daily_avg["date"]]
    prices = daily_avg["price"].tolist()
    indices = list(range(len(prices)))

    plt.bar(indices, prices, color="green")
    plt.title(f"Daily Average Electricity Prices - {region}")
    plt.xlabel("Date")
    plt.ylabel(f"Average Price ({df['unit'].iloc[0]})")
    plt.grid(True, True)

    # Set x-axis labels
    if len(dates) > 10:
        step = max(1, len(dates) // 10)
        x_indices = [i for i in range(0, len(dates), step)]
        x_labels = [dates[i] for i in x_indices]
        plt.xticks(x_indices, x_labels)
    else:
        plt.xticks(indices, dates)

    plt.show()

    print(f"\n📈 Daily averages for {region}")
    print("=" * 30)
    for _, row in daily_avg.iterrows():
        print(f"{row['date']}: {row['price']:.2f} {df['unit'].iloc[0]}")
    print()


def create_hourly_analysis_chart(df: pd.DataFrame, region: str) -> None:
    """
    Create terminal charts showing hourly price patterns and duck curves.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for holiday detection and display
    """
    if df.empty:
        print(f"No data available for {region}")
        return
    
    analyzer = HourlyPriceAnalyzer(region)
    results = analyzer.analyze_hourly_patterns(df)
    
    if not results:
        print("No hourly analysis results available")
        return
    
    unit = df['unit'].iloc[0] if not df.empty else 'EUR/MWh'
    
    # Chart 1: Workday vs Non-workday comparison
    if 'workday' in results and 'non_workday' in results:
        plt.clear_data()
        plt.clear_figure()
        
        workday_stats = results['workday']
        nonworkday_stats = results['non_workday']
        
        hours = list(range(24))
        workday_prices = [workday_stats[workday_stats['hour'] == h]['mean'].iloc[0] if not workday_stats[workday_stats['hour'] == h].empty else 0 for h in hours]
        nonworkday_prices = [nonworkday_stats[nonworkday_stats['hour'] == h]['mean'].iloc[0] if not nonworkday_stats[nonworkday_stats['hour'] == h].empty else 0 for h in hours]
        
        plt.plot(hours, workday_prices, label="Workdays", color="red", marker="hd")
        plt.plot(hours, nonworkday_prices, label="Weekends/Holidays", color="blue", marker="hd") 
        
        start_date = df["timestamp"].min().strftime("%Y-%m-%d")
        end_date = df["timestamp"].max().strftime("%Y-%m-%d")
        country_name = _get_country_name(region)
        plt.title(f"Duck Curve Analysis - {country_name} ({start_date} to {end_date})")
        plt.xlabel("Hour of Day")
        plt.ylabel(f"Average Price ({unit})")
        
        # Set x-axis to show every 4 hours
        plt.xticks(list(range(0, 24, 4)), [f"{h}:00" for h in range(0, 24, 4)])
        
        # Add grid lines
        plt.grid(True, True)
        
        plt.show()
    
    # Analysis summary
    print(f"\n🦆 Duck Curve Analysis for {region}")
    print("=" * 50)
    
    comparison = analyzer.compare_workday_vs_nonworkday(results)
    
    if 'workday_features' in comparison:
        features = comparison['workday_features']
        print(f"📊 WORKDAY PATTERN")
        print(f"Morning peak:   {features.get('morning_peak_hour', 'N/A')}:00 at {features.get('morning_peak_price', 0):.1f} {unit}")
        print(f"Midday minimum: {features.get('midday_min_hour', 'N/A')}:00 at {features.get('midday_min_price', 0):.1f} {unit}")
        print(f"Evening peak:   {features.get('evening_peak_hour', 'N/A')}:00 at {features.get('evening_peak_price', 0):.1f} {unit}")
        print(f"Duck depth:     {features.get('duck_depth', 0):.1f} {unit}")
        print(f"Evening ramp:   {features.get('evening_ramp', 0):.1f} {unit}")
        
        # Calculate duck curve strength
        from ..analysis.hourly import detect_duck_curve_strength
        strength = detect_duck_curve_strength(df, region)
        print(f"Duck strength:  {strength:.3f} (0=flat, 1=pronounced)")
    
    if 'non_workday_features' in comparison:
        features = comparison['non_workday_features']
        print(f"\n📊 WEEKEND/HOLIDAY PATTERN")
        print(f"Morning peak:   {features.get('morning_peak_hour', 'N/A')}:00 at {features.get('morning_peak_price', 0):.1f} {unit}")
        print(f"Midday minimum: {features.get('midday_min_hour', 'N/A')}:00 at {features.get('midday_min_price', 0):.1f} {unit}")
        print(f"Evening peak:   {features.get('evening_peak_hour', 'N/A')}:00 at {features.get('evening_peak_price', 0):.1f} {unit}")
        print(f"Duck depth:     {features.get('duck_depth', 0):.1f} {unit}")
        print(f"Evening ramp:   {features.get('evening_ramp', 0):.1f} {unit}")
    
    if 'differences' in comparison:
        diff = comparison['differences']
        print(f"\n🔍 WORKDAY vs WEEKEND/HOLIDAY DIFFERENCES")
        print(f"Duck depth difference:  {diff.get('duck_depth_diff', 0):.1f} {unit}")
        print(f"Evening ramp difference: {diff.get('evening_ramp_diff', 0):.1f} {unit}")
        print(f"Price range difference:  {diff.get('price_range_diff', 0):.1f} {unit}")
    
    print()


def create_hourly_workday_chart(df: pd.DataFrame, region: str) -> None:
    """
    Create terminal chart showing workday hourly patterns only.
    
    Args:
        df: DataFrame with columns ['timestamp', 'price', 'unit']
        region: Region code for holiday detection and display
    """
    if df.empty:
        print(f"No data available for {region}")
        return
    
    analyzer = HourlyPriceAnalyzer(region)
    results = analyzer.analyze_hourly_patterns(df)
    
    if 'workday' not in results:
        print("No workday data available for analysis")
        return
    
    workday_stats = results['workday']
    unit = df['unit'].iloc[0] if not df.empty else 'EUR/MWh'
    
    plt.clear_data()
    plt.clear_figure()
    
    hours = list(range(24))
    prices = [workday_stats[workday_stats['hour'] == h]['mean'].iloc[0] if not workday_stats[workday_stats['hour'] == h].empty else 0 for h in hours]
    
    plt.plot(hours, prices, marker="hd", color="cyan")
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    country_name = _get_country_name(region)
    plt.title(f"Workday Duck Curve - {country_name} ({start_date} to {end_date})")
    plt.xlabel("Hour of Day")
    plt.ylabel(f"Average Price ({unit})")
    
    # Set x-axis to show every 2 hours for better detail
    plt.xticks(list(range(0, 24, 2)), [f"{h}:00" for h in range(0, 24, 2)])
    
    # Add grid lines
    plt.grid(True, True)
    
    plt.show()
    
    # Show duck curve features
    features = analyzer.detect_duck_curve_features(workday_stats)
    print(f"\n🦆 Workday Duck Curve Features for {region}")
    print("=" * 40)
    print(f"Morning peak:   {features.get('morning_peak_hour', 'N/A')}:00 at {features.get('morning_peak_price', 0):.1f} {unit}")
    print(f"Midday minimum: {features.get('midday_min_hour', 'N/A')}:00 at {features.get('midday_min_price', 0):.1f} {unit}")
    print(f"Evening peak:   {features.get('evening_peak_hour', 'N/A')}:00 at {features.get('evening_peak_price', 0):.1f} {unit}")
    print(f"Duck depth:     {features.get('duck_depth', 0):.1f} {unit}")
    print(f"Evening ramp:   {features.get('evening_ramp', 0):.1f} {unit}")
    print()


def create_terminal_duck_factor_chart(
    duck_factors_df: pd.DataFrame,
    region: str,
    window_days: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """
    Create a terminal-based duck factor time series chart.
    
    Args:
        duck_factors_df: DataFrame with duck factor time series
        region: Region code for the chart title
        window_days: Rolling window size used
        width: Chart width (defaults to terminal width)
        height: Chart height (defaults to auto)
    """
    if duck_factors_df.empty:
        print("No duck factor data available for charting")
        return
    
    # Clear any previous plots
    plt.clear_data()
    plt.clear_figure()
    
    # Prepare data
    dates = pd.to_datetime(duck_factors_df['date'])
    factors = duck_factors_df['duck_factor'].tolist()
    indices = list(range(len(factors)))
    
    # Create the plot
    plt.plot(indices, factors, marker="hd", color="yellow")
    
    # Set title and labels
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")
    country_name = _get_country_name(region)
    title = f"Duck Factor Evolution ({window_days}d window) - {country_name} ({start_date} to {end_date})"
    
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Duck Factor (0-1)")
    
    # Set chart size if specified
    if width:
        plt.plotsize(width, height or 15)
    elif height:
        plt.plotsize(None, height)
    
    # Configure x-axis labels
    total_points = len(dates)
    if total_points > 15:
        step = max(1, total_points // 8)
        x_indices = [i for i in range(0, total_points, step)]
        x_labels = [dates.iloc[i].strftime("%Y-%m") for i in x_indices]
        plt.xticks(x_indices, x_labels)
    else:
        x_labels = [d.strftime("%Y-%m-%d") for d in dates]
        plt.xticks(indices, x_labels)
    
    # Display the chart
    plt.show()
    
    # Print summary statistics
    print(f"\nDuck Factor Summary:")
    print(f"Average:    {np.mean(factors) if factors else 0:.3f}")
    print(f"Range:      {(np.max(factors) - np.min(factors)) if factors else 0:.3f}")
    print(f"Trend:      {_calculate_simple_trend(factors)}")
    print(f"Data points: {len(factors)}")
    print()


def create_png_duck_factor_chart(
    duck_factors_df: pd.DataFrame,
    region: str,
    output_path: str,
    window_days: int,
    width: int = 12,
    height: int = 6
) -> None:
    """
    Create a PNG duck factor time series chart.
    
    Args:
        duck_factors_df: DataFrame with duck factor time series
        region: Region code for the chart title
        output_path: Path to save the PNG file
        window_days: Rolling window size used
        width: Figure width in inches
        height: Figure height in inches
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for PNG output. Please install with: pip install matplotlib>=3.7.0")
    
    if duck_factors_df.empty:
        print("No duck factor data available for charting")
        return
    
    # Create figure and axis
    fig, ax = mpl_plt.subplots(figsize=(width, height))
    
    # Prepare data
    dates = pd.to_datetime(duck_factors_df['date'])
    factors = duck_factors_df['duck_factor']
    
    # Plot the time series
    ax.plot(dates, factors, marker='o', markersize=3, linewidth=1.5, color='#FF6B35', alpha=0.8)
    
    # Add trend line
    if len(factors) > 3:
        try:
            x_numeric = dates.astype(np.int64) / 10**9
            trend_coef = np.polyfit(x_numeric, factors, 1)
            trend_line = np.polyval(trend_coef, x_numeric)
            ax.plot(dates, trend_line, '--', color='#2E86AB', linewidth=2, alpha=0.7, label=f'Trend')
            ax.legend(loc='upper right')
        except Exception:
            pass
    
    # Set title and labels
    start_date = dates.min().strftime("%Y-%m-%d")
    end_date = dates.max().strftime("%Y-%m-%d")
    country_name = _get_country_name(region)
    ax.set_title(f"Duck Factor Evolution ({window_days}d window) - {country_name} ({start_date} to {end_date})", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Duck Factor (0-1)", fontsize=12)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 240)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    
    # Rotate x-axis labels for better readability
    mpl_plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show 0-1 range
    ax.set_ylim(0, 1)
    
    # Adjust layout and save
    mpl_plt.tight_layout()
    mpl_plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mpl_plt.close(fig)
    
    print(f"Duck factor chart saved to: {output_path}")


def create_png_seasonal_duck_chart(
    seasonal_data: Dict[str, Any],
    region: str,
    output_path: str,
    width: int = 12,
    height: int = 8
) -> None:
    """
    Create a PNG chart showing seasonal duck factor patterns.
    
    Args:
        seasonal_data: Seasonal analysis results from RollingDuckAnalyzer
        region: Region code for the chart title
        output_path: Path to save the PNG file
        width: Figure width in inches
        height: Figure height in inches
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for PNG output. Please install with: pip install matplotlib>=3.7.0")
    
    if 'seasonal_patterns' not in seasonal_data or seasonal_data['seasonal_patterns'].empty:
        print("No seasonal data available for charting")
        return
    
    # Create subplots for different seasonal views
    fig, ((ax1, ax2), (ax3, ax4)) = mpl_plt.subplots(2, 2, figsize=(width, height))
    
    seasonal_df = seasonal_data['seasonal_patterns']
    monthly_df = seasonal_data.get('monthly_patterns', pd.DataFrame())
    
    # 1. Seasonal averages (top-left)
    if not seasonal_df.empty:
        seasons = seasonal_df['season']
        means = seasonal_df['mean']
        stds = seasonal_df['std']
        
        bars = ax1.bar(seasons, means, yerr=stds, capsize=5, color=['#A8DADC', '#457B9D', '#1D3557', '#F1C40F'])
        ax1.set_title('Duck Factor by Season', fontweight='bold')
        ax1.set_ylabel('Duck Factor')
        ax1.set_ylim(0, max(means) * 1.2)
        ax1.grid(True, alpha=0.3)
    
    # 2. Monthly patterns (top-right)
    if not monthly_df.empty:
        months = monthly_df['month']
        month_means = monthly_df['mean']
        month_stds = monthly_df['std']
        
        ax2.errorbar(months, month_means, yerr=month_stds, marker='o', capsize=3, 
                    color='#E63946', linewidth=2, markersize=6)
        ax2.set_title('Duck Factor by Month', fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Duck Factor')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(month_means) * 1.2)
    
    # 3. Seasonal range visualization (bottom-left)
    if not seasonal_df.empty:
        seasonal_range = seasonal_data.get('seasonal_range', 0)
        peak_season = seasonal_data.get('peak_season', 'Unknown')
        low_season = seasonal_data.get('low_season', 'Unknown')
        
        # Simple text display of key metrics
        ax3.text(0.1, 0.8, f"Peak Season: {peak_season}", fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.6, f"Low Season: {low_season}", fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.4, f"Seasonal Range: {seasonal_range:.3f}", fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.2, f"Peak/Low Ratio: {means.max()/means.min():.2f}x" if means.min() > 0 else "Peak/Low Ratio: N/A", 
                fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Seasonal Metrics', fontweight='bold')
        ax3.axis('off')
    
    # 4. Season comparison polar plot (bottom-right)
    if not seasonal_df.empty and len(seasonal_df) == 4:
        # Create a simple radar-style comparison
        angles = np.linspace(0, 2 * np.pi, len(seasons), endpoint=False).tolist()
        values = means.tolist()
        
        # Complete the circle
        angles += angles[:1]
        values += values[:1]
        
        ax4 = mpl_plt.subplot(2, 2, 4, projection='polar')
        ax4.plot(angles, values, 'o-', linewidth=2, color='#2A9D8F')
        ax4.fill(angles, values, alpha=0.25, color='#2A9D8F')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(seasons)
        ax4.set_title('Seasonal Duck Factor Pattern', fontweight='bold', pad=20)
        ax4.set_ylim(0, max(values) * 1.1)
    
    # Overall title
    country_name = _get_country_name(region)
    fig.suptitle(f'Seasonal Duck Factor Analysis - {country_name}', fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    mpl_plt.tight_layout()
    mpl_plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mpl_plt.close(fig)
    
    print(f"Seasonal duck factor chart saved to: {output_path}")


def create_terminal_negative_pricing_chart(
    df: pd.DataFrame,
    region: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    near_zero_threshold: float = 5.0,
) -> None:
    """
    Create a terminal chart showing negative pricing patterns by hour of day.
    
    Args:
        df: DataFrame with price data
        region: Region code
        width: Chart width
        height: Chart height
        near_zero_threshold: Threshold for near-zero pricing (EUR/MWh)
    """
    analyzer = NegativePricingAnalyzer(region)
    metrics = analyzer.analyze_negative_pricing_patterns(df, near_zero_threshold)
    
    if not metrics.hourly_breakdown:
        print("No data available for negative pricing analysis")
        return
    
    # Clear any previous plots
    plt.clear_data()
    plt.clear_figure()
    
    # Prepare hourly data
    hours = list(range(24))
    negative_percentages = [metrics.hourly_breakdown.get(h, {}).get('negative_percentage', 0) for h in hours]
    near_zero_percentages = [metrics.hourly_breakdown.get(h, {}).get('near_zero_percentage', 0) for h in hours]
    
    # Create the plot
    plt.bar(hours, negative_percentages, marker="fhd", color="red", label="Negative Prices")
    plt.bar(hours, near_zero_percentages, marker="fhd", color="orange", label="Near-Zero Prices")
    
    # Set title and labels
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    country_name = _get_country_name(region)
    title = f"Negative Pricing Patterns - {country_name} ({start_date} to {end_date})"
    subtitle = f"Negative: <0 EUR/MWh, Near-Zero: ≤{near_zero_threshold} EUR/MWh"
    
    plt.title(title)
    plt.xlabel("Hour of Day")
    plt.ylabel("Percentage of Hours (%)")
    
    # Set chart size if specified
    if width:
        plt.plotsize(width, height or 15)
    elif height:
        plt.plotsize(None, height)
    
    # Configure x-axis
    plt.xticks(range(0, 24, 4), [f"{h}:00" for h in range(0, 24, 4)])
    
    # Display the chart
    plt.show()
    
    # Print threshold information and summary
    print(f"Thresholds: {subtitle}")
    print(f"\nNegative Pricing Summary:")
    print(f"Total negative hours: {metrics.negative_hours} ({metrics.negative_percentage:.1f}%)")
    print(f"Average hours/day: {metrics.avg_hours_per_day:.1f}")
    print(f"Max consecutive: {metrics.max_consecutive_hours} hours")
    print()


def create_terminal_negative_pricing_timechart(
    df: pd.DataFrame,
    region: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    near_zero_threshold: float = 5.0,
    aggregation_level: str = "daily",
) -> None:
    """
    Create a terminal timechart showing aggregated hours with negative/near-zero prices.
    
    Args:
        df: DataFrame with price data
        region: Region code
        width: Chart width
        height: Chart height
        near_zero_threshold: Threshold for near-zero pricing (EUR/MWh)
        aggregation_level: Aggregation level - "daily", "weekly", or "monthly"
    """
    from ..analysis.negative_pricing import calculate_aggregated_hours_timeseries
    
    # Calculate aggregated hours
    aggregated_data = calculate_aggregated_hours_timeseries(df, aggregation_level, near_zero_threshold)
    
    if aggregated_data.empty:
        print("No data available for negative pricing timechart")
        return
    
    # Clear any previous plots
    plt.clear_data()
    plt.clear_figure()
    
    # Prepare data for plotting
    # Use numeric indices for x-axis instead of date strings to avoid plotext date parsing issues
    x_values = list(range(len(aggregated_data)))
    negative_hours = aggregated_data['negative_hours'].tolist()
    near_zero_hours = aggregated_data['near_zero_hours'].tolist()
    
    # Create the plot - use line plot for time series
    plt.plot(x_values, negative_hours, marker="hd", color="red", label="Negative Prices (<0)")
    plt.plot(x_values, near_zero_hours, marker="hd", color="orange", label=f"Near-Zero (≤{near_zero_threshold})")
    
    # Set title and labels based on aggregation level
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    country_name = _get_country_name(region)
    
    aggregation_labels = {
        "daily": ("Daily Hours", "Hours per Day"),
        "weekly": ("Weekly Hours", "Hours per Week"),
        "monthly": ("Monthly Hours", "Hours per Month")
    }
    
    title_text, ylabel_text = aggregation_labels.get(aggregation_level, ("Daily Hours", "Hours per Day"))
    title = f"{title_text} with Negative/Near-Zero Prices - {country_name}"
    subtitle = f"Negative: <0 EUR/MWh, Near-Zero: ≤{near_zero_threshold} EUR/MWh"
    
    plt.title(title)
    plt.xlabel("Time Period")
    plt.ylabel(ylabel_text)
    
    # Set chart size if specified
    if width:
        plt.plotsize(width, height or 15)
    elif height:
        plt.plotsize(None, height)
    
    # Configure x-axis to show meaningful time period labels
    time_periods = aggregated_data['time_period']
    if aggregation_level == "daily":
        labels = time_periods.dt.strftime('%Y-%m-%d').tolist()
    elif aggregation_level == "weekly":
        labels = time_periods.dt.strftime('%Y-W%U').tolist()  # Year-Week format
    else:  # monthly
        labels = time_periods.dt.strftime('%Y-%m').tolist()
    
    num_periods = len(labels)
    if num_periods > 20:
        # Show every nth label to avoid crowding
        step = max(1, num_periods // 10)
        label_indices = list(range(0, num_periods, step))
        label_values = [labels[i] for i in label_indices]
        plt.xticks(label_indices, label_values)
    else:
        # Show all labels if there aren't too many
        plt.xticks(x_values, labels)
    
    # Display the chart
    plt.show()
    
    # Print threshold information and summary
    print(f"Thresholds: {subtitle}")
    print(f"\n{title_text.split()[0]} Summary:")
    
    period_unit = aggregation_level.replace("ly", "").replace("y", "")  # daily->day, weekly->week, monthly->month
    print(f"Average negative hours/{period_unit}: {aggregated_data['negative_hours'].mean():.1f}")
    print(f"Average near-zero hours/{period_unit}: {aggregated_data['near_zero_hours'].mean():.1f}")
    print(f"Max negative hours in a {period_unit}: {aggregated_data['negative_hours'].max()}")
    print(f"Max near-zero hours in a {period_unit}: {aggregated_data['near_zero_hours'].max()}")
    print(f"{aggregation_labels[aggregation_level][0].split()[0]} periods with any negative prices: {(aggregated_data['negative_hours'] > 0).sum()}")
    print()


def create_png_negative_pricing_chart(
    df: pd.DataFrame,
    region: str,
    output_path: str,
    width: int = 12,
    height: int = 8,
    near_zero_threshold: float = 5.0,
) -> None:
    """
    Create a comprehensive PNG chart for negative pricing analysis.
    
    Args:
        df: DataFrame with price data
        region: Region code
        output_path: Path to save PNG file
        width: Figure width
        height: Figure height
        near_zero_threshold: Threshold for near-zero pricing (EUR/MWh)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for PNG output. Please install with: pip install matplotlib>=3.7.0")
    
    analyzer = NegativePricingAnalyzer(region)
    metrics = analyzer.analyze_negative_pricing_patterns(df, near_zero_threshold)
    seasonal_data = analyzer.analyze_seasonal_patterns(df)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = mpl_plt.subplots(2, 2, figsize=(width, height))
    
    # 1. Hourly patterns (top-left)
    if metrics.hourly_breakdown:
        hours = list(range(24))
        negative_pct = [metrics.hourly_breakdown.get(h, {}).get('negative_percentage', 0) for h in hours]
        near_zero_pct = [metrics.hourly_breakdown.get(h, {}).get('near_zero_percentage', 0) for h in hours]
        
        ax1.bar(hours, negative_pct, alpha=0.7, color='red', label='Negative Prices')
        ax1.bar(hours, near_zero_pct, alpha=0.7, color='orange', label='Near-Zero Prices')
        ax1.set_title('Negative Pricing by Hour of Day', fontweight='bold')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Percentage of Hours (%)')
        ax1.set_xticks(range(0, 24, 4))
        ax1.set_xticklabels([f"{h}:00" for h in range(0, 24, 4)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Monthly patterns (top-right)
    if metrics.monthly_breakdown:
        months = list(range(1, 13))
        monthly_negative = [metrics.monthly_breakdown.get(m, {}).get('avg_hours_per_day', 0) for m in months]
        
        ax2.plot(months, monthly_negative, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax2.set_title('Average Negative Pricing Hours per Day by Month', fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Hours per Day')
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax2.grid(True, alpha=0.3)
    
    # 3. Solar potential vs current (bottom-left)
    if 'yearly_summary' in seasonal_data:
        current_avg = seasonal_data['yearly_summary']['avg_current_hours_per_day']
        potential_avg = seasonal_data['yearly_summary']['avg_theoretical_max_hours_per_day']
        
        categories = ['Current\nActual', 'Theoretical\nMaximum']
        values = [current_avg, potential_avg]
        colors = ['#FF6B35', '#2A9D8F']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.7)
        ax3.set_title('Current vs Theoretical Maximum', fontweight='bold')
        ax3.set_ylabel('Hours per Day')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}h', ha='center', va='bottom', fontweight='bold')
    
    # 4. Progress by month (bottom-right)
    if seasonal_data and len(seasonal_data) > 1:  # Exclude yearly_summary
        month_nums = []
        progress_values = []
        
        for month, data in seasonal_data.items():
            if month != 'yearly_summary' and 'progress_metrics' in data:
                if 'error' not in data['progress_metrics']:
                    month_nums.append(month)
                    progress_values.append(data['progress_metrics']['progress_percentage'])
        
        if month_nums:
            ax4.plot(month_nums, progress_values, marker='s', linewidth=2, markersize=6, color='#E63946')
            ax4.set_title('Progress Toward Maximum by Month', fontweight='bold')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Progress (%)')
            ax4.set_xticks(range(1, 13))
            ax4.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax4.set_ylim(0, 100)
            ax4.grid(True, alpha=0.3)
    
    # Overall title
    country_name = _get_country_name(region)
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    fig.suptitle(f'Negative Pricing Analysis - {country_name} ({start_date} to {end_date})\nNegative: <0 EUR/MWh, Near-Zero: ≤{near_zero_threshold} EUR/MWh', 
                fontsize=16, fontweight='bold')
    
    # Adjust layout and save
    mpl_plt.tight_layout()
    mpl_plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mpl_plt.close(fig)
    
    print(f"Negative pricing analysis chart saved to: {output_path}")


def create_png_negative_pricing_timechart(
    df: pd.DataFrame,
    region: str,
    output_path: str,
    width: int = 12,
    height: int = 6,
    near_zero_threshold: float = 5.0,
    aggregation_level: str = "daily",
) -> None:
    """
    Create a PNG timechart showing aggregated hours with negative/near-zero prices.
    
    Args:
        df: DataFrame with price data
        region: Region code
        output_path: Path to save PNG file
        width: Figure width
        height: Figure height
        near_zero_threshold: Threshold for near-zero pricing (EUR/MWh)
        aggregation_level: Aggregation level - "daily", "weekly", or "monthly"
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for PNG output. Please install with: pip install matplotlib>=3.7.0")
    
    from ..analysis.negative_pricing import calculate_aggregated_hours_timeseries
    
    # Calculate aggregated hours
    aggregated_data = calculate_aggregated_hours_timeseries(df, aggregation_level, near_zero_threshold)
    
    if aggregated_data.empty:
        print("No data available for negative pricing timechart")
        return
    
    # Create figure
    fig, ax = mpl_plt.subplots(1, 1, figsize=(width, height))
    
    # Plot data
    ax.plot(aggregated_data['time_period'], aggregated_data['negative_hours'], 
            marker='o', linewidth=2, markersize=4, color='#DC143C', 
            label='Negative Prices (<0 EUR/MWh)')
    ax.plot(aggregated_data['time_period'], aggregated_data['near_zero_hours'], 
            marker='s', linewidth=2, markersize=4, color='#FF8C00', 
            label=f'Near-Zero (≤{near_zero_threshold} EUR/MWh)')
    
    # Set title and labels based on aggregation level
    country_name = _get_country_name(region)
    start_date = df["timestamp"].min().strftime("%Y-%m-%d")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d")
    
    aggregation_labels = {
        "daily": ("Daily Hours", "Hours per Day"),
        "weekly": ("Weekly Hours", "Hours per Week"),
        "monthly": ("Monthly Hours", "Hours per Month")
    }
    
    title_text, ylabel_text = aggregation_labels.get(aggregation_level, ("Daily Hours", "Hours per Day"))
    
    ax.set_title(f'{title_text} with Negative/Near-Zero Prices - {country_name}\n({start_date} to {end_date})', 
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Time Period', fontweight='bold')
    ax.set_ylabel(ylabel_text, fontweight='bold')
    
    # Configure x-axis based on aggregation level
    num_periods = len(aggregated_data)
    
    if aggregation_level == "daily":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Auto-adjust x-axis labels based on date range
        if num_periods > 365:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif num_periods > 90:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        elif num_periods > 30:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, num_periods // 10)))
    elif aggregation_level == "weekly":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if num_periods > 52:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
    else:  # monthly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if num_periods > 24:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    mpl_plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Add summary text box
    avg_negative = aggregated_data['negative_hours'].mean()
    avg_near_zero = aggregated_data['near_zero_hours'].mean()
    max_negative = aggregated_data['negative_hours'].max()
    max_near_zero = aggregated_data['near_zero_hours'].max()
    periods_with_negative = (aggregated_data['negative_hours'] > 0).sum()
    
    period_unit = aggregation_level.replace("ly", "").replace("y", "")  # daily->day, weekly->week, monthly->month
    summary_text = f'Avg negative: {avg_negative:.1f} hrs/{period_unit}\nAvg near-zero: {avg_near_zero:.1f} hrs/{period_unit}\nMax negative: {max_negative} hrs\n{title_text.split()[0]} periods with negative: {periods_with_negative}'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    # Adjust layout and save
    mpl_plt.tight_layout()
    mpl_plt.savefig(output_path, dpi=300, bbox_inches='tight')
    mpl_plt.close(fig)
    
    print(f"Negative pricing timechart saved to: {output_path}")


def _calculate_simple_trend(values: list) -> str:
    """Calculate a simple trend description for a list of values."""
    if len(values) < 3:
        return "Insufficient data"
    
    try:
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 0.001:
            return "Stable"
        elif slope > 0.01:
            return "Strong upward"
        elif slope > 0.005:
            return "Moderate upward"
        elif slope > 0:
            return "Slight upward"
        elif slope < -0.01:
            return "Strong downward"
        elif slope < -0.005:
            return "Moderate downward"
        else:
            return "Slight downward"
    except Exception:
        return "Unable to calculate"
