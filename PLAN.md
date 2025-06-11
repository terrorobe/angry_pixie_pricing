# Angry Pixie Pricing - Development Plan

## Project Overview
European electricity price analysis tool with terminal-based charting and smart meter cost calculations.

## Current Status ✅

### Core Infrastructure
- [x] CLI framework with Click (chart, calculate, sources, clear-cache commands)
- [x] Abstract data source architecture with caching
- [x] EnergyCharts.info API integration
- [x] CSV + gzip cache system with data source attribution
- [x] Month-based caching for past months, day-based for current month
- [x] Cache filename format: `{source}_{region}_{period}.csv.gz`
- [x] Case-normalized region codes (always uppercase)

### Visualization System
- [x] Terminal charting with plotext
- [x] Line charts for hourly prices with braille markers
- [x] Daily average bar charts
- [x] Statistical summary tables with price statistics
- [x] Chart type options: `line`, `daily`, `summary`, `all`
- [x] Customizable chart dimensions (`--width`, `--height`)
- [x] Smart x-axis labeling with automatic density adjustment

### Code Quality
- [x] Git repository initialized
- [x] Comprehensive .gitignore for generated artifacts
- [x] Black code formatting applied
- [x] Clean git history (no generated files committed)

## ✅ COMPLETED: Holiday-Aware Duck Curve Analysis 🦆

### Objective ✅ ACHIEVED
Successfully implemented comprehensive duck curve analysis that differentiates between workdays and weekends/holidays, revealing distinct European electricity market patterns driven by economic activity and renewable energy integration.

### Research Completed ✅
- [x] Identified Python `holidays` library as optimal data source
- [x] Verified accuracy against authoritative sources (TimeAndDate.com)
- [x] Confirmed coverage for Western European countries (DE, FR, NL, BE, AT, CH)
- [x] Validated workday vs weekend/holiday classification strategy

### Implementation Completed ✅

#### Phase 1: Holiday Infrastructure ✅ COMPLETE
- [x] Add `holidays>=0.34` to requirements.txt
- [x] Create `src/angry_pixie_pricing/analysis/day_types.py` module
- [x] Implement `classify_day_type(timestamp, country_code)` function
- [x] Add utility functions for workday/weekend/holiday detection
- [x] Test day type classification with real electricity data

#### Phase 2: Hourly Analysis Engine ✅ COMPLETE
- [x] Create `src/angry_pixie_pricing/analysis/hourly.py` module
- [x] Implement `analyze_hourly_patterns(df, region)` function
- [x] Calculate average prices by hour-of-day for each day type
- [x] Generate statistical confidence intervals for hourly averages
- [x] Add duck curve feature detection (morning peak, midday dip, evening ramp)
- [x] Implement duck curve strength scoring (0-1 scale)
- [x] Add negative price analysis by time-of-day

#### Phase 3: Duck Curve Visualization ✅ COMPLETE
- [x] Create `create_hourly_analysis_chart()` in `charts/terminal.py`
- [x] Implement side-by-side workday vs weekend/holiday comparison
- [x] Add chart options: `--chart-type hourly`, `--chart-type hourly-workday`
- [x] Create duck curve visualization highlighting morning/evening peaks
- [x] Add comprehensive CLI integration with enhanced help text
- [x] Validate with real German summer data showing textbook duck curves

### 🎯 Validation Results
- **Germany July 2023**: Duck curve strength 0.849 (highly pronounced)
- **Workday pattern**: 6AM peak → 12PM solar dip → 6PM evening peak  
- **Weekend pattern**: Even more pronounced with negative midday prices (-52.0 EUR/MWh)
- **Key insight**: Weekends show 38.2 EUR/MWh deeper duck curves than workdays

### Phase 4: Advanced Features (Future)
- [ ] Regional holiday support (German states, etc.)
- [ ] Bridge day detection (holidays creating long weekends)
- [ ] School holiday integration for enhanced analysis
- [ ] Seasonal stratification options (summer vs winter patterns)
- [ ] Export functionality for hourly analysis data

## Architecture Decisions Made

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
- **Phase 1**: Simple hour-of-day averaging, ignore DST complexity initially
- **Future**: UTC-based analysis or local solar time approximation
- **Rationale**: Duck curve analysis more important than DST precision initially

### Visualization Philosophy
- **Terminal-first**: Fast, dependency-light, SSH-friendly
- **Human-readable**: Clear labels, appropriate scaling, informative summaries
- **Progressive complexity**: Basic charts first, advanced analysis optional

## Known Technical Debt

### Code Quality
- [ ] Add comprehensive unit test suite
- [ ] Implement mypy type checking (currently has import-untyped warnings)
- [ ] Add integration tests for API data fetching
- [ ] Add docstring examples for all public functions

### Performance
- [ ] Implement async data fetching for multiple regions
- [ ] Add progress bars for long data fetches
- [ ] Optimize large dataset handling for multi-year analysis

### Features
- [ ] Smart meter cost calculation implementation (currently TODO)
- [ ] File export options for charts (PNG, SVG)
- [ ] More data sources beyond EnergyCharts.info
- [ ] Configuration file support for default settings

## Session Handoff Notes

### For Next Session
1. Start with implementing the holidays dependency and day classification utilities
2. The cache system is solid - focus on analysis features
3. All chart infrastructure exists - new chart types can be added easily
4. Region normalization is handled - always use uppercase in cache filenames

### Key Files to Know
- `src/angry_pixie_pricing/data/base.py` - Abstract caching and data source framework
- `src/angry_pixie_pricing/data/energy_charts.py` - EnergyCharts.info API implementation  
- `src/angry_pixie_pricing/charts/terminal.py` - All visualization functions
- `src/angry_pixie_pricing/main.py` - CLI command definitions

### Development Workflow
1. Use TodoWrite tool to track session progress
2. Commit frequently with descriptive messages
3. Update this PLAN.md file when major features complete
4. Run `black src/` before committing for consistent formatting

## Long-term Vision

### Advanced Analysis Features
- Renewable energy correlation analysis
- Cross-country price comparison
- Demand response event detection
- Grid congestion price spike analysis
- Carbon intensity correlation with pricing

### Data Integration
- Multiple electricity market APIs
- Weather data integration for renewable correlation
- Industrial production indices for demand analysis
- Carbon pricing data for environmental analysis

### User Experience
- Web dashboard for non-technical users
- Automated report generation
- Alert system for price anomalies
- Historical trend analysis and forecasting

---

*Last updated: Session where CSV caching was implemented and holiday analysis planned*
*Next priority: Holiday-aware hourly analysis implementation*