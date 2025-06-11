# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📋 Project Status & Development Plan

**IMPORTANT**: See `PLAN.md` for current development status, roadmap, and next session tasks. Always read PLAN.md when starting a new session to understand what to work on next.

## Project Overview

Angry Pixie Pricing is a Python tool for analyzing electricity prices in European markets. It supports two main modes:
1. **Chart Generation**: Create hourly electricity price charts for specific regions and time spans
2. **Usage Analysis**: Calculate average electricity costs based on smart meter data (15-minute increments)

## Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Commands

### Running the Application
```bash
# Generate price charts
angry-pixie chart --region DE --start-date 2024-01-01 --end-date 2024-01-31 --output chart.png

# Calculate costs from smart meter data
angry-pixie calculate --usage-data meter_data.csv --region DE --start-date 2024-01-01 --end-date 2024-01-31
```

### Development Commands
```bash
# Run tests
pytest tests/

# Code formatting
black src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/
```

## Architecture

### Project Structure
- `src/angry_pixie_pricing/` - Main package
  - `data/` - Data acquisition and processing modules
  - `charts/` - Chart generation for price visualization
  - `analysis/` - Smart meter data analysis and cost calculations
  - `utils/` - Utility functions and helpers
  - `main.py` - CLI entry point
- `tests/` - Test files
- `data/` - Data storage (raw and processed)

### Key Components
- **CLI Interface**: Click-based command-line tool with chart and calculate commands
- **Data Processing**: Handles European electricity market data and smart meter readings
- **Visualization**: Creates charts for hourly price analysis
- **Cost Analysis**: Processes 15-minute smart meter data for cost calculations

## 📊 Data Visualization Guidelines

**CRITICAL**: All charts must display raw data points connected by straight lines only, without curve interpolation unless explicitly requested by the user.

### Chart Requirements
- Use `plt.plot()` with markers to connect discrete data points with straight lines
- No curve smoothing, spline interpolation, or bezier curves between hourly data points
- Each data point represents actual hourly electricity prices
- Lines connect adjacent time periods with simple straight segments
- Use plotext markers (braille, dot, etc.) to show individual data points clearly

### Rationale
- Electricity prices are discrete hourly values, not continuous functions
- Straight line connections preserve data integrity while showing trends
- Curve interpolation can misrepresent actual market data and create misleading patterns
- Users need to see exact price points connected by simple linear segments
- Duck curve analysis requires precise hourly values with clear connections between periods