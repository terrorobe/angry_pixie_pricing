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

### Python Environment
The project uses a Python virtual environment located at `./venv/` in the project root.

**CRITICAL**: Always activate the virtual environment before running any commands:
```bash
# Activate virtual environment (required for all operations)
source venv/bin/activate

# Verify environment is active (should show project path)
which python
which angry-pixie
```

### Installation
```bash
# If venv doesn't exist, create it first:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running Commands
All commands must be run with the virtual environment activated:
```bash
# ALWAYS start with this
source venv/bin/activate

# Then run commands
angry-pixie chart --region DE --start-date 2024-07
angry-pixie negative-pricing --region AT --start-date 2020
angry-pixie duck-factor --region DE --start-date 2019 --end-date 2024
```

### Testing & Debugging
```bash
# Activate venv first
source venv/bin/activate

# Run tests
python test_script.py

# Direct module execution
python -m src.angry_pixie_pricing.main --help
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
angry-pixie chart --region DE --start-date 2024-07                    # July 2024
angry-pixie chart --region DE --start-date 2024                       # Whole year 2024
angry-pixie chart --region DE --start-date 2024-07-15                 # July 15 to today

# Generate PNG charts with auto-naming
angry-pixie chart --region DE --start-date 2024-07 --output auto
# Creates: images/prices_de_20240701_20240731.png

# Duck curve analysis  
angry-pixie chart --region DE --start-date 2024-07 --chart-type hourly --output duck_curve
```

#### Duck Factor Analysis (Rolling Window)
```bash
# Basic duck factor evolution over 6 years
angry-pixie duck-factor --region DE --start-date 2019 --end-date 2024

# Seasonal duck factor analysis
angry-pixie duck-factor --region DE --start-date 2024 --chart-type seasonal --output analysis

# Multi-window comparison (7d, 30d, 90d windows)
angry-pixie duck-factor --region DE --start-date 2020 --chart-type multi-window --output multi

# Custom window and step size
angry-pixie duck-factor --region DE --start-date 2024 --window 14d --step 3d
```

#### Negative Pricing Analysis
```bash
# Analyze negative pricing patterns with solar potential
angry-pixie negative-pricing --region DE --start-date 2024

# Custom threshold and PNG output
angry-pixie negative-pricing --region DE --start-date 2024-06 --threshold 10.0 --output negative_analysis

# Multi-year negative pricing trends
angry-pixie negative-pricing --region DE --start-date 2020 --end-date 2024
```

#### Smart Meter Cost Calculation
```bash
# Calculate costs from smart meter data
angry-pixie calculate --usage-data meter_data.csv --region DE --start-date 2024-07
```

### Development Commands
```bash
# ALWAYS activate venv first
source venv/bin/activate

# Run tests
pytest tests/

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Troubleshooting

### Python Environment Issues

**Error: `command not found: python` or `command not found: angry-pixie`**
```bash
# Solution: Activate the virtual environment
source venv/bin/activate
```

**Error: `ModuleNotFoundError: No module named 'pandas'` (or other dependencies)**
```bash
# Solution: Virtual environment not activated or dependencies not installed
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**Error: `externally-managed-environment` when using pip**
```bash
# Solution: Use the virtual environment instead of system pip
source venv/bin/activate  # This uses venv's pip, not system pip
pip install -e .
```

**Testing Fixes:**
```bash
# Create a simple test script to verify environment
source venv/bin/activate
python -c "import pandas; import numpy; print('Environment OK')"

# Test specific functionality
angry-pixie --help
angry-pixie chart --region DE --start-date 2024-01 --end-date 2024-01
```

### Environment Verification
```bash
# Check virtual environment is active
source venv/bin/activate
which python     # Should show: /Users/terrorobe/source/angry_pixie_pricing/venv/bin/python
which angry-pixie # Should show: /Users/terrorobe/source/angry_pixie_pricing/venv/bin/angry-pixie

# Check installed packages
pip list | grep -E "(pandas|numpy|click|plotext)"
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

### Examples
- Title: "Electricity Spot Prices - Germany (2023-07-01 to 2023-07-31)"
- Filter example: "Duck Curve Analysis - Germany (2023-07-01 to 2023-07-31)"
- Workday filter: "Workday Duck Curve - Germany (2023-07-01 to 2023-07-31)"
- Axis: "Price (EUR/MWh)" with automatic grid lines

### Rationale
- Electricity prices are discrete hourly values, not continuous functions
- Straight line connections preserve data integrity while showing trends
- Curve interpolation can misrepresent actual market data and create misleading patterns
- Users need complete context about what data they're viewing
- Grid lines help users quickly estimate price levels and ranges

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