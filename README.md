# Angry Pixie Pricing

European Electricity Price Analysis Tool - A comprehensive Python tool for analyzing electricity prices in European markets.

## Features

- **Chart Generation**: Create hourly electricity price charts for specific regions and time spans
- **Duck Curve Analysis**: Rolling window analysis of renewable energy impact on pricing patterns
- **Negative Pricing Analysis**: Track hours of negative/near-zero pricing with solar potential modeling
- **Usage Analysis**: Calculate average electricity costs based on smart meter data (15-minute increments)

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run a command
uv run angry-pixie chart --region DE --start-date 2024-07
```

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed usage instructions and development guidelines.