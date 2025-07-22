"""Visualization tools for load pattern analysis."""

from pathlib import Path
from typing import Any

import pandas as pd
import plotext as plt_term


class LoadPatternCharts:
    """Generate charts for load pattern analysis."""

    @staticmethod
    def _generate_coverage_title(base_title: str, region: str, coverage_info: dict[str, Any]) -> str:
        """Generate a title showing actual data coverage with precise dates."""
        start_date = coverage_info["actual_start_date"]
        end_date = coverage_info["actual_end_date"]

        # Format dates as YYYY-MM-DD
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        coverage_str = f"({start_str} to {end_str})"

        return f"{base_title} - {region.upper()} {coverage_str}"

    @staticmethod
    def plot_hourly_peaks_evolution(
        peaks_df: pd.DataFrame,
        region: str,
        coverage_info: dict[str, Any],
        percentile: float = 95.0,
        output_file: str | Path | None = None,
    ) -> None:
        """
        Plot how peak loads for each hour have evolved over time.

        Args:
            peaks_df: DataFrame with hourly peaks by year
            region: Region code for chart title
            coverage_info: Dict with actual data coverage information
            percentile: Percentile used for peak calculation (for title display)
            output_file: Optional file path for saving chart as image
        """
        # Generate title with actual coverage and percentile info
        base_title = f"Hourly Peak Load Evolution (P{percentile:g})"
        title = LoadPatternCharts._generate_coverage_title(base_title, region, coverage_info)

        # Pivot data for plotting
        peaks_pivot = peaks_df.pivot_table(index="hour", columns="year", values="peak_load", aggfunc="mean")

        if output_file:
            # High-resolution matplotlib chart
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

                # Use energy hue progression palette for years
                from angry_pixie_pricing.utils.color_palettes import get_year_colors

                colors = get_year_colors(len(peaks_pivot.columns), "energy")

                # Plot each year
                for i, year in enumerate(peaks_pivot.columns):
                    ax.plot(
                        peaks_pivot.index,
                        peaks_pivot[year] / 1000,  # Convert to GW
                        marker="o",
                        markersize=4,
                        linewidth=2,
                        label=str(year),
                        color=colors[i],
                    )

                ax.set_xlabel("Hour of Day", fontsize=12, fontweight="bold")
                ax.set_ylabel("Peak Load (GW)", fontsize=12, fontweight="bold")
                ax.set_title(
                    title,
                    fontsize=14,
                    fontweight="bold",
                    pad=20,
                )

                # Format x-axis
                ax.set_xticks(range(0, 24, 2))
                ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
                ax.grid(True, alpha=0.3)

                # Legend
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Year", title_fontsize=10, fontsize=9)

                plt.tight_layout()

                # Save high-resolution image
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"📊 High-resolution chart saved: {output_path}")

            except ImportError:
                print("⚠ matplotlib not available, using terminal chart only")
                output_file = None

        # Terminal chart
        plt_term.clear_data()
        plt_term.clear_color()

        # Plot each year as a separate line
        for year in peaks_pivot.columns:
            year_data = peaks_pivot[year].dropna()
            if len(year_data) > 0:
                plt_term.plot(
                    year_data.index.tolist(),
                    (year_data / 1000).tolist(),  # Convert to GW
                    marker="hd",
                    label=str(year),
                )

        # Chart formatting
        plt_term.title(title)
        plt_term.xlabel("Hour of Day")
        plt_term.ylabel("Peak Load (GW)")

        # Set x-axis labels
        plt_term.xticks(list(range(0, 24, 3)), [f"{h:02d}:00" for h in range(0, 24, 3)])

        plt_term.grid(True)
        plt_term.show()

    @staticmethod
    def plot_duck_curve_evolution(
        duck_metrics_df: pd.DataFrame,
        region: str,
        output_file: str | Path | None = None,
    ) -> None:
        """
        Plot duck curve intensity evolution over time.

        Args:
            duck_metrics_df: DataFrame with duck curve metrics by year
            region: Region code for chart title
            output_file: Optional file path for saving chart as image
        """
        if output_file:
            # High-resolution matplotlib chart
            try:
                import matplotlib.pyplot as plt

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10), dpi=300)

                years = duck_metrics_df["year"]

                # Get consistent colors from energy palette
                from angry_pixie_pricing.utils.color_palettes import get_year_colors

                metric_colors = get_year_colors(4, "energy")  # 4 different metrics

                # Duck intensity
                ax1.plot(years, duck_metrics_df["duck_intensity_pct"], marker="o", linewidth=2, color=metric_colors[0])
                ax1.set_title("Duck Curve Intensity", fontweight="bold")
                ax1.set_ylabel("Intensity (%)", fontweight="bold")
                ax1.grid(True, alpha=0.3)

                # Peak loads - use energy palette for different peak types
                peak_colors = get_year_colors(3, "energy")  # 3 peak types
                ax2.plot(
                    years,
                    duck_metrics_df["morning_peak_mw"] / 1000,
                    marker="s",
                    linewidth=2,
                    label="Morning Peak",
                    color=peak_colors[0],
                )
                ax2.plot(
                    years,
                    duck_metrics_df["evening_peak_mw"] / 1000,
                    marker="^",
                    linewidth=2,
                    label="Evening Peak",
                    color=peak_colors[1],
                )
                ax2.plot(
                    years,
                    duck_metrics_df["midday_min_mw"] / 1000,
                    marker="v",
                    linewidth=2,
                    label="Midday Min",
                    color=peak_colors[2],
                )
                ax2.set_title("Peak vs Midday Load", fontweight="bold")
                ax2.set_ylabel("Load (GW)", fontweight="bold")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Midday suppression
                ax3.plot(
                    years,
                    duck_metrics_df["midday_suppression_pct"],
                    marker="d",
                    linewidth=2,
                    color=metric_colors[1],
                )
                ax3.set_title("Midday Demand Suppression", fontweight="bold")
                ax3.set_ylabel("Suppression (%)", fontweight="bold")
                ax3.set_xlabel("Year", fontweight="bold")
                ax3.grid(True, alpha=0.3)

                # Peak-to-trough ratio
                ax4.plot(years, duck_metrics_df["peak_trough_ratio"], marker="h", linewidth=2, color=metric_colors[2])
                ax4.set_title("Peak-to-Trough Ratio", fontweight="bold")
                ax4.set_ylabel("Ratio", fontweight="bold")
                ax4.set_xlabel("Year", fontweight="bold")
                ax4.grid(True, alpha=0.3)

                fig.suptitle(f"Duck Curve Analysis - {region.upper()}", fontsize=16, fontweight="bold")

                plt.tight_layout()

                # Save high-resolution image
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"📊 Duck curve analysis saved: {output_path}")

            except ImportError:
                print("⚠ matplotlib not available, using terminal chart only")

        # Terminal chart - duck intensity
        plt_term.clear_data()
        plt_term.clear_color()

        years = duck_metrics_df["year"].tolist()
        intensity = duck_metrics_df["duck_intensity_pct"].tolist()

        plt_term.plot(years, intensity, marker="hd")
        plt_term.title(f"Duck Curve Intensity Evolution - {region.upper()}")
        plt_term.xlabel("Year")
        plt_term.ylabel("Duck Intensity (%)")
        plt_term.grid(True)
        plt_term.show()

    @staticmethod
    def plot_peak_migration_heatmap(
        peaks_df: pd.DataFrame,
        region: str,
        reference_year: int | None = None,
        output_file: str | Path | None = None,
    ) -> None:
        """
        Plot a heatmap showing peak load changes by hour and year.

        Args:
            peaks_df: DataFrame with hourly peaks by year
            region: Region code for chart title
            reference_year: Reference year for comparison (defaults to earliest)
            output_file: Optional file path for saving chart as image
        """
        # Pivot data
        peaks_pivot = peaks_df.pivot_table(index="hour", columns="year", values="peak_load", aggfunc="mean")

        if reference_year is None:
            reference_year = peaks_pivot.columns.min()

        # Calculate percentage changes from reference year
        if reference_year not in peaks_pivot.columns:
            reference_year = peaks_pivot.columns.min()

        reference_data = peaks_pivot[reference_year]
        changes_df = peaks_pivot.div(reference_data, axis=0).subtract(1).multiply(100)
        changes_df = changes_df.drop(columns=[reference_year], errors="ignore")

        if output_file:
            # High-resolution matplotlib heatmap
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

                # Create heatmap
                sns.heatmap(
                    changes_df.T,  # Transpose to have years on y-axis
                    annot=True,
                    fmt=".1f",
                    cmap="RdBu_r",
                    center=0,
                    cbar_kws={"label": "Change from Reference Year (%)"},
                    ax=ax,
                )

                ax.set_title(
                    f"Peak Load Changes by Hour - {region.upper()}\n(Reference: {reference_year})",
                    fontsize=14,
                    fontweight="bold",
                    pad=20,
                )
                ax.set_xlabel("Hour of Day", fontweight="bold")
                ax.set_ylabel("Year", fontweight="bold")

                # Format x-axis
                hour_labels = [f"{h:02d}:00" if h % 3 == 0 else "" for h in range(24)]
                ax.set_xticklabels(hour_labels)

                plt.tight_layout()

                # Save high-resolution image
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"📊 Peak migration heatmap saved: {output_path}")

            except ImportError:
                print("⚠ matplotlib/seaborn not available, using terminal display")

        # Terminal summary
        plt_term.clear_data()
        plt_term.clear_color()

        # Show average change by hour across all years
        avg_change = changes_df.mean(axis=1)

        plt_term.plot(avg_change.index.tolist(), avg_change.tolist(), marker="hd")
        plt_term.title(f"Average Peak Load Change by Hour - {region.upper()}\n(vs {reference_year})")
        plt_term.xlabel("Hour of Day")
        plt_term.ylabel("Average Change (%)")
        plt_term.xticks(list(range(0, 24, 3)), [f"{h:02d}:00" for h in range(0, 24, 3)])
        plt_term.grid(True)
        plt_term.show()
