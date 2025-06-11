"""Terminal-based charting using plotext."""

import plotext as plt
import pandas as pd
from typing import Optional
from ..analysis.hourly import HourlyPriceAnalyzer


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

    # Create the plot
    plt.plot(indices, prices, marker="braille", color="cyan")

    # Set title
    if title is None:
        start_date = df["timestamp"].min().strftime("%Y-%m-%d")
        end_date = df["timestamp"].max().strftime("%Y-%m-%d")
        unit = df["unit"].iloc[0] if not df.empty else "EUR/MWh"
        title = f"Electricity Spot Prices - {region} ({start_date} to {end_date})"

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
        
        plt.plot(hours, workday_prices, label="Workdays", color="red", marker="braille")
        plt.plot(hours, nonworkday_prices, label="Weekends/Holidays", color="blue", marker="braille") 
        
        plt.title(f"Duck Curve Analysis - {region} (Workdays vs Weekends/Holidays)")
        plt.xlabel("Hour of Day")
        plt.ylabel(f"Average Price ({unit})")
        
        # Set x-axis to show every 4 hours
        x_labels = [f"{h}:00" for h in range(0, 24, 4)]
        plt.xticks(range(0, 24, 4), x_labels)
        
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
    
    plt.plot(hours, prices, marker="braille", color="cyan")
    plt.title(f"Workday Duck Curve - {region}")
    plt.xlabel("Hour of Day")
    plt.ylabel(f"Average Price ({unit})")
    
    # Set x-axis to show every 2 hours for better detail
    x_labels = [f"{h}:00" for h in range(0, 24, 2)]
    plt.xticks(range(0, 24, 2), x_labels)
    
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
