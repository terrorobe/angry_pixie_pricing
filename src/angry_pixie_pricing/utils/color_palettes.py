"""Shared color palettes for consistent chart styling."""

# Energy-themed color palette with proper hue progression (Blue 240° → Red 0°)
# Optimized for year-over-year line plots with up to 10 years of data
ENERGY_HUE_10 = [
    "#0B4D6A",  # Deep Blue (Hydro/Ocean)
    "#1E6091",  # Blue (Water)
    "#2980B9",  # Bright Blue (Wind)
    "#16A085",  # Teal (Clean Energy)
    "#27AE60",  # Green (Renewables)
    "#8BC34A",  # Light Green (Biomass)
    "#FFC107",  # Yellow (Solar)
    "#FF9500",  # Orange (Natural Gas)
    "#E74C3C",  # Red-Orange (Conventional)
    "#C0392B",  # Deep Red (Coal/Peak)
]

# Alternative palettes for different use cases
SCIENTIFIC_QUALITATIVE = [
    "#332288",
    "#117733",
    "#44AA99",
    "#88CCEE",
    "#DDCC77",
    "#CC6677",
    "#AA4499",
    "#882255",
    "#661100",
    "#999933",
]

MATPLOTLIB_TAB10_HUE_SORTED = [
    "#d62728",
    "#ff7f0e",
    "#bcbd22",
    "#2ca02c",
    "#17becf",
    "#1f77b4",
    "#9467bd",
    "#e377c2",
    "#8c564b",
    "#7f7f7f",
]


def get_year_colors(n_years: int, palette: str = "energy") -> list[str]:
    """
    Get colors for year-over-year line plots.

    Args:
        n_years: Number of years to color
        palette: Palette to use ("energy", "scientific", "tab10")

    Returns:
        List of hex color codes
    """
    palettes = {"energy": ENERGY_HUE_10, "scientific": SCIENTIFIC_QUALITATIVE, "tab10": MATPLOTLIB_TAB10_HUE_SORTED}

    colors = palettes.get(palette, ENERGY_HUE_10)

    # If fewer years than available colors, spread them across the full range
    if n_years <= len(colors):
        # Use evenly spaced indices to get full color range
        indices = [int(i * (len(colors) - 1) / max(1, n_years - 1)) for i in range(n_years)]
        return [colors[i] for i in indices]
    # More years than colors - cycle through
    return [colors[i % len(colors)] for i in range(n_years)]
