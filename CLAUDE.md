# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📋 Project Status & Development Plan

**IMPORTANT**: See `PLAN.md` for current development status, roadmap, and next session tasks. Always read PLAN.md when starting a new session to understand what to work on next.

## Project Overview

Angry Pixie Pricing is a comprehensive Python tool for analyzing electricity prices in European markets. It supports multiple analysis modes:

1. **Chart Generation**: Create hourly electricity price charts for specific regions and time spans
2. **Duck Curve Analysis**: Rolling window analysis of renewable energy impact on pricing patterns
3. **Negative Pricing Analysis**: Track hours of negative/near-zero pricing with solar potential modeling
4. **Usage Analysis**: Calculate average electricity costs based on smart meter data (15-minute increments)

## Development Setup

### Modern Python with uv
This project uses `uv` for fast, reliable Python package management.

### Installation
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies and create virtual environment
uv sync

# Install package in development mode with dev dependencies
uv sync --dev
```

### Running Commands
```bash
# Run commands directly with uv (no activation needed!)
uv run angry-pixie chart --region DE --start-date 2024-07

# Or activate the virtual environment for traditional usage
source .venv/bin/activate
angry-pixie chart --region DE --start-date 2024-07
```

### Testing & Debugging
```bash
# Run tests with uv
uv run pytest tests/

# Run specific test file
uv run python test_script.py

# Direct module execution
uv run python -m src.angry_pixie_pricing.main --help

# Interactive Python with all dependencies
uv run ipython
```

## Commands

### Running the Application

#### Flexible Date Input
The tool supports flexible date formats for convenience:
- **YYYY**: Whole year (e.g., `2024` = Jan 1 - Dec 31, 2024)
- **YYYY-MM**: Whole month (e.g., `2024-07` = July 1-31, 2024)  
- **YYYY-MM-DD**: Specific date (e.g., `2024-07-15`)
- **No end-date**: Defaults to today

#### Basic Price Charts
```bash
# Generate terminal charts (flexible date formats)
uv run angry-pixie chart --region DE --start-date 2024-07              # July 2024
uv run angry-pixie chart --region DE --start-date 2024                 # Whole year 2024
uv run angry-pixie chart --region DE --start-date 2024-07-15           # July 15 to today

# Generate PNG charts with auto-naming
uv run angry-pixie chart --region DE --start-date 2024-07 --output auto
# Creates: images/prices_de_20240701_20240731.png

## 📂 PNG Filename Format
Auto-generated PNG filenames follow this pattern:
```
{chart_type}_{region}_{start_date}_{end_date}[_{window}][_{suffix}].png
```

**Examples:**
- `prices_de_20240701_20240731.png` - German prices for July 2024
- `duck-factor_de_20190101_20241231_30d_timeseries.png` - Duck factor analysis with 30-day window
- `load-peaks_at_20200101_20241231_hourly-evolution.png` - Austrian load peaks 2020-2024
- `load-peaks_at_20200101_20241231_hourly-evolution-p99.png` - Austrian load peaks with p99 filter
- `negative-pricing_fr_20230101_20231231.png` - French negative pricing analysis

**Components:**
- **chart_type**: `prices`, `duck-factor`, `load-peaks`, `negative-pricing`
- **region**: Country code (lowercase): `de`, `at`, `fr`
- **dates**: YYYYMMDD format (no hyphens)
- **window** (optional): `7d`, `30d`, `90d` for rolling analyses
- **suffix** (optional): `timeseries`, `seasonal`, `workday`, `hourly-evolution`, `duck-curve`, `peak-migration`
- **filter** (optional for load-peaks): `p99`, `p95`, `p90` for percentile filtering

All files saved to `images/` directory relative to project root.

# Duck curve analysis  
uv run angry-pixie chart --region DE --start-date 2024-07 --chart-type hourly --output duck_curve
```

#### Duck Factor Analysis (Rolling Window)
```bash
# Basic duck factor evolution over 6 years
uv run angry-pixie duck-factor --region DE --start-date 2019 --end-date 2024

# Seasonal duck factor analysis
uv run angry-pixie duck-factor --region DE --start-date 2024 --chart-type seasonal --output analysis

# Multi-window comparison (7d, 30d, 90d windows)
uv run angry-pixie duck-factor --region DE --start-date 2020 --chart-type multi-window --output multi

# Custom window and step size
uv run angry-pixie duck-factor --region DE --start-date 2024 --window 14d --step 3d
```

#### Negative Pricing Analysis
```bash
# Analyze negative pricing patterns with solar potential
uv run angry-pixie negative-pricing --region DE --start-date 2024

# Custom threshold and PNG output
uv run angry-pixie negative-pricing --region DE --start-date 2024-06 --threshold 10.0 --output negative_analysis

# Multi-year negative pricing trends
uv run angry-pixie negative-pricing --region DE --start-date 2020 --end-date 2024
```

#### Smart Meter Cost Calculation
```bash
# Calculate costs from smart meter data
uv run angry-pixie calculate --usage-data meter_data.csv --region DE --start-date 2024-07
```

### Development Commands
```bash
# Run tests
uv run pytest tests/

# Code Quality
```bash
# Format and check code (must have zero issues before work is complete)
uv run ruff format . && uv run ruff check . --fix && uv run mypy src/
```

**CRITICAL**: Work is only complete when both ruff and mypy report zero issues. Always run before committing.

## Troubleshooting

### uv Installation Issues

**Error: `command not found: uv`**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew on macOS
brew install uv
```

**Error: `ModuleNotFoundError` when running commands**
```bash
# Solution: Always use 'uv run' prefix or activate the environment
uv run angry-pixie --help

# Or activate and use traditionally
source .venv/bin/activate
angry-pixie --help
```

**Testing Fixes:**
```bash
# Verify uv installation
uv --version

# Test environment setup
uv run python -c "import pandas; import numpy; print('Environment OK')"

# Test specific functionality
uv run angry-pixie --help
uv run angry-pixie chart --region DE --start-date 2024-01
```

### Environment Verification
```bash
# Check Python version
uv run python --version  # Should show Python 3.12.x

# List installed packages
uv pip list

# Show project dependencies
uv tree
```

## Architecture

### Project Structure
- `src/angry_pixie_pricing/` - Main package
  - `data/` - Data acquisition and processing modules
  - `charts/` - Chart generation for price visualization (terminal and PNG)
  - `analysis/` - Advanced electricity market analysis modules
    - `hourly.py` - Duck curve and hourly pattern analysis
    - `rolling_duck.py` - Multi-window duck factor evolution analysis
    - `negative_pricing.py` - Negative pricing patterns with solar potential modeling
    - `day_types.py` - Workday/holiday classification utilities
  - `utils/` - Utility functions and helpers
    - `date_parser.py` - Flexible date format parsing
    - `filename_generator.py` - Smart output file naming with project root anchoring
    - `cli_options.py` - Reusable CLI option decorators
  - `main.py` - CLI entry point with multiple analysis commands
- `tests/` - Test files
- `images/` - Generated chart outputs (gitignored)
- `data/` - Data storage (raw and processed)

### Key Components
- **CLI Interface**: Click-based command-line tool with multiple analysis commands (`chart`, `duck-factor`, `negative-pricing`, `calculate`)
- **Flexible Date Parsing**: Supports YYYY, YYYY-MM, YYYY-MM-DD formats with smart defaults
- **Data Processing**: Handles European electricity market data and smart meter readings
- **Advanced Analytics**: 
  - Rolling duck factor analysis with trend detection
  - Negative pricing analysis with solar potential modeling
  - Seasonal pattern detection and year-over-year comparisons
- **Visualization**: Creates both terminal and high-resolution PNG charts
- **Smart Output Management**: Auto-generated filenames with project-root anchoring

## 📊 Data Visualization Guidelines

**CRITICAL**: All charts must display raw data points connected by straight lines only, without curve interpolation unless explicitly requested by the user.

### Chart Requirements
**Terminal Charts (plotext):**
- Use `plt.plot()` with markers to connect discrete data points with straight lines
- Use `marker="hd"` (default plotext marker) for clear data point visibility
- No curve smoothing, spline interpolation, or bezier curves between hourly data points

**PNG Charts (matplotlib):**
- Use `ax.plot()` with small markers ('o', 's') and appropriate line width
- High-resolution output (300 DPI) with publication-quality formatting
- Proper axis labeling, grid lines, and legends
- No curve smoothing, spline interpolation, or bezier curves between hourly data points

**Universal Requirements:**
- Each data point represents actual hourly electricity prices
- Lines connect adjacent time periods with simple straight segments

### Chart Content Requirements
- **Data Context**: All charts must include region, time range, and any applied filters in titles/labels
- **Price Axis**: Clearly labeled price axis with appropriate units (EUR/MWh, etc.)
- **Grid Lines**: Enable both horizontal and vertical grid lines for easy reading
- **Filter Information**: Show when data is filtered (e.g., "Workdays Only", "Holidays Excluded")
- **Time Range**: Always display the exact date range being analyzed
- **⚠️ CRITICAL - Incomplete Data Coverage**: When actual data coverage differs from requested time range, this MUST be clearly indicated in the chart title and/or warning messages

### Data Coverage Documentation

**CRITICAL REQUIREMENT**: Chart titles must always show the actual data range covered, not the requested range.

#### Implementation Requirements:
1. **Title Shows Actual Coverage**: Always display the exact date range of available data
   - `"Load Analysis - DE (2020-2023)"` (not "2020-2024" if 2024 is unavailable)
   - `"Duck Curve - AT (2024-01-01 to 2024-06-15)"` (show actual end date)

2. **Granularity Adjustment**: When requested granularity isn't available, adjust gracefully
   - If yearly data unavailable → use monthly granularity
   - If monthly data unavailable → use daily granularity
   - Document the granularity used: `"Daily Load Patterns - DE (2024-07-01 to 2024-07-15)"`

3. **Console Information**: Display coverage info before charts
   - `"📊 Data Coverage: 2020-01-01 to 2023-12-31 (3 complete years)"`
   - `"📊 Using daily granularity (monthly aggregation unavailable)"`

#### Examples
- Complete coverage: `"Electricity Spot Prices - Germany (2023-07-01 to 2023-07-31)"`
- Partial coverage: `"Electricity Spot Prices - Germany (2023-07-01 to 2023-07-28)"`
- Granularity adjustment: `"Daily Load Patterns - Germany (2024-01-01 to 2024-01-31)"`

### Rationale
- Electricity prices are discrete hourly values, not continuous functions
- Straight line connections preserve data integrity while showing trends
- Curve interpolation can misrepresent actual market data and create misleading patterns
- Users need complete context about what data they're viewing
- Grid lines help users quickly estimate price levels and ranges

## 🏗️ Architecture Decisions

### Caching Strategy
- **Format**: CSV + gzip compression for portability and human readability
- **Naming**: `{datasource}_{region}_{period}.csv.gz` (e.g., `energycharts_DE_2023-11.csv.gz`)
- **Granularity**: Whole months for past data, flexible periods for current month
- **Benefits**: Universal compatibility, easy inspection, clear data attribution

### Day Type Classification
- **Strategy**: Treat weekends and public holidays equivalently for electricity analysis
- **Rationale**: Both represent reduced economic activity and different consumption patterns
- **Implementation**: Three categories: `workday`, `weekend`, `holiday` (combine latter two for analysis)

### DST Handling Approach
- **Current**: Simple hour-of-day averaging, ignore DST complexity initially
- **Future**: UTC-based analysis or local solar time approximation
- **Rationale**: Duck curve analysis more important than DST precision initially

### Visualization Philosophy
- **Terminal-first**: Fast, dependency-light, SSH-friendly
- **Human-readable**: Clear labels, appropriate scaling, informative summaries
- **Progressive complexity**: Basic charts first, advanced analysis optional

## 📈 Analysis Capabilities

### Duck Factor Analysis
- **Rolling Windows**: Configurable window sizes (7d, 30d, 90d) with customizable step sizes
- **Trend Detection**: Statistical trend analysis with R-squared confidence metrics
- **Seasonal Patterns**: Automatic detection of seasonal variations in duck curve strength
- **Year-over-Year Tracking**: Progress metrics showing renewable energy adoption
- **Multi-Window Comparison**: Compare different time scales simultaneously

### Negative Pricing Analysis
- **Solar Potential Modeling**: Estimates theoretical maximum negative pricing hours based on:
  - EU PVGIS solar irradiation data
  - Global Solar Atlas regional data
  - Current solar capacity and grid flexibility factors
- **Progress Metrics**: Track how close each region is to solar saturation
- **Hourly Patterns**: Identify peak negative pricing hours and seasonal variations
- **Capacity Scenarios**: Model impacts of 2x, 5x, 10x current solar capacity

### Data Sources & Attribution
- **Solar Irradiation**: EU PVGIS, Global Solar Atlas, Copernicus Climate Data Store
- **Solar Capacity**: SolarPower Europe, IRENA Global Energy Transformation statistics
- **Grid Flexibility**: Estimated based on hydro storage, interconnection, demand response capabilities
- **Note**: All data sources are approximate and compiled for modeling purposes

## 🚀 Future Development Roadmap

### Advanced Analysis Features
- Renewable energy correlation analysis
- Cross-country price comparison dashboards
- Demand response event detection
- Grid congestion price spike analysis
- Carbon intensity correlation with pricing patterns

### Data Integration Expansion
- Multiple electricity market APIs (ENTSO-E, Nord Pool, etc.)
- Weather data integration for renewable correlation analysis
- Industrial production indices for demand pattern analysis
- Real-time solar capacity tracking from national grid operators

### Smart Meter Integration
- Complete implementation of cost calculation functionality
- 15-minute increment processing with dynamic pricing
- Demand response optimization recommendations

## Recent Updates & Fixes

### Latest Session (Bug Fixes & Usability)
- ✅ **Fixed Critical Error**: Resolved `'Columns not found: False, True'` error in negative pricing analysis
- ✅ **Eliminated Warnings**: Fixed numpy empty slice warnings in seasonal calculations
- ✅ **Enhanced Charts**: Added threshold information display (e.g., "Negative: <0 EUR/MWh, Near-Zero: ≤5.0 EUR/MWh")
- ✅ **Documentation**: Added comprehensive Python environment setup and troubleshooting guide
- ✅ **System Stability**: All core features now working reliably with clear user feedback

### Current System Status
All major analysis commands are working correctly:
```bash
uv run angry-pixie chart --region DE --start-date 2024-07                    # ✅ Working
uv run angry-pixie duck-factor --region DE --start-date 2019 --end-date 2024 # ✅ Working  
uv run angry-pixie negative-pricing --region AT --start-date 2020            # ✅ Fixed & Working
```