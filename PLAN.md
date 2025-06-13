# Angry Pixie Pricing - Development Plan

## Project Overview
Comprehensive European electricity price analysis tool featuring duck curve evolution tracking, negative pricing analysis with solar potential modeling, and smart meter cost calculations.

## Current Status ✅

### Core Infrastructure
- [x] CLI framework with Click (chart, duck-factor, negative-pricing, calculate, sources, clear-cache commands)
- [x] Abstract data source architecture with caching
- [x] EnergyCharts.info API integration
- [x] CSV + gzip cache system with data source attribution
- [x] Month-based caching for past months, day-based for current month
- [x] Cache filename format: `{source}_{region}_{period}.csv.gz`
- [x] Case-normalized region codes (always uppercase)
- [x] Flexible date parsing (YYYY, YYYY-MM, YYYY-MM-DD) with smart defaults
- [x] Project root anchoring for consistent output paths
- [x] Centralized CLI option definitions with decorators

### Visualization System
- [x] Terminal charting with plotext for fast analysis
- [x] High-resolution PNG charts with matplotlib (300 DPI)
- [x] Line charts for hourly prices with braille markers
- [x] Daily average bar charts
- [x] Statistical summary tables with price statistics
- [x] Duck curve visualizations (workday vs weekend/holiday patterns)
- [x] Rolling duck factor time series charts
- [x] Seasonal duck factor analysis with polar plots
- [x] Negative pricing analysis with solar potential comparisons
- [x] Chart type options: `line`, `daily`, `summary`, `hourly`, `hourly-workday`, `all`
- [x] Customizable chart dimensions (`--width`, `--height`)
- [x] Smart x-axis labeling with automatic density adjustment
- [x] Auto-generated descriptive filenames with organized output structure

### Code Quality
- [x] Git repository initialized
- [x] Comprehensive .gitignore for generated artifacts
- [x] Black code formatting applied
- [x] Clean git history (no generated files committed)

## ✅ COMPLETED: Advanced Market Analysis Suite 🦆⚡

### Objective ✅ ACHIEVED
Successfully implemented comprehensive electricity market analysis suite including:
1. **Holiday-aware duck curve analysis** with workday/weekend differentiation
2. **Rolling duck factor evolution** tracking renewable energy impact over multiple years
3. **Negative pricing analysis** with solar potential modeling and grid saturation estimates

### Duck Curve Analysis ✅ COMPLETE

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

### Rolling Duck Factor Analysis ✅ COMPLETE

#### Phase 1: Rolling Window Implementation ✅ COMPLETE
- [x] Create `src/angry_pixie_pricing/analysis/rolling_duck.py` module
- [x] Implement configurable rolling windows (7d, 30d, 90d) with custom step sizes
- [x] Add multi-window comparative analysis capabilities
- [x] Implement statistical trend detection with R-squared confidence metrics

#### Phase 2: Temporal Pattern Analysis ✅ COMPLETE
- [x] Add seasonal pattern detection (Winter/Spring/Summer/Fall)
- [x] Implement year-over-year comparison and growth tracking
- [x] Add inflection point detection for significant market changes
- [x] Create volatility trend analysis for market stability assessment

#### Phase 3: Duck Factor Visualization ✅ COMPLETE
- [x] Create terminal and PNG time series charts for duck factor evolution
- [x] Add seasonal analysis charts with polar plots and monthly breakdowns
- [x] Implement multi-window comparison visualizations
- [x] Add comprehensive CLI command `duck-factor` with multiple chart types

### Negative Pricing Analysis ✅ COMPLETE

#### Phase 1: Solar Potential Modeling ✅ COMPLETE
- [x] Create `src/angry_pixie_pricing/analysis/negative_pricing.py` module
- [x] Integrate EU PVGIS solar irradiation data for 13 European countries
- [x] Add Global Solar Atlas and Copernicus Climate Data Store attribution
- [x] Implement current solar capacity estimates and grid flexibility factors

#### Phase 2: Saturation Analysis ✅ COMPLETE
- [x] Calculate theoretical maximum hours of negative pricing per region/month
- [x] Model multiple capacity scenarios (current, 2x, 5x, 10x solar capacity)
- [x] Implement progress metrics showing advancement toward grid saturation
- [x] Add seasonal variation analysis for solar potential

#### Phase 3: Negative Pricing Visualization ✅ COMPLETE
- [x] Create comprehensive 4-panel PNG charts (hourly, monthly, current vs max, progress)
- [x] Add terminal charts for negative pricing patterns by hour of day
- [x] Implement CLI command `negative-pricing` with threshold customization
- [x] Add detailed analysis summaries with regional context

### 🎯 Validation Results
- **Germany 2019-2025**: Duck factor 0.663 average, strong seasonal variation (Spring peak, Winter low)
- **Renewable Progress**: 4.7% average annual growth in duck curve strength
- **Solar Potential**: Comprehensive modeling showing current vs theoretical maximum hours
- **Regional Insights**: Grid flexibility differences between hydro-rich (Norway 0.95) vs constrained (Italy 0.5) countries

### CLI Usability Improvements ✅ COMPLETE
- [x] Flexible date parsing supporting YYYY, YYYY-MM, YYYY-MM-DD formats
- [x] Smart end-date defaults (automatically defaults to today when omitted)
- [x] Project root anchoring for consistent image output regardless of execution directory
- [x] Centralized CLI option definitions using decorators to eliminate code duplication
- [x] Auto-generated descriptive filenames with organized `images/` directory structure

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

## Next Development Priorities

### High Priority Features
- [ ] Smart meter cost calculation implementation (currently TODO placeholder)
- [ ] Real-time API integration for live solar irradiation data (replacing static estimates)
- [ ] Cross-regional comparative analysis dashboards
- [ ] Historical trend forecasting models

### Code Quality & Testing
- [ ] Add comprehensive unit test suite for all analysis modules
- [ ] Implement mypy type checking (currently has import-untyped warnings)
- [ ] Add integration tests for API data fetching and analysis pipelines
- [ ] Add docstring examples for all public functions
- [ ] Performance benchmarking for multi-year analysis workflows

### Advanced Analytics
- [ ] Correlation analysis between duck curves and weather patterns
- [ ] Grid congestion pricing spike detection algorithms
- [ ] Carbon intensity correlation with pricing patterns
- [ ] Machine learning models for price anomaly detection

### Data Integration Expansion
- [ ] More electricity market APIs (ENTSO-E, Nord Pool, etc.)
- [ ] Weather data integration for renewable correlation analysis
- [ ] Industrial production indices for demand pattern analysis
- [ ] Real-time solar capacity tracking from national grid operators

## ✅ COMPLETED: Bug Fixes & Usability Improvements 🔧📊

### Recent Session Accomplishments
Successfully resolved critical bugs and enhanced user experience:

#### 🐛 Bug Fixes Completed
1. **Negative Pricing Analysis Error**: Fixed pandas groupby error where boolean mask was incorrectly used as column selector
2. **Missing Numpy Import**: Added numpy import to main.py for seasonal analysis calculations
3. **F-string Syntax Error**: Fixed incorrect syntax in duck factor summary calculations
4. **Empty Slice Warnings**: Eliminated numpy warnings by adding proper empty list checking

#### 📈 Usability Enhancements Completed
1. **Threshold Display in Charts**: Both terminal and PNG charts now show exact criteria used (e.g., "Negative: <0 EUR/MWh, Near-Zero: ≤5.0 EUR/MWh")
2. **Enhanced Documentation**: Added comprehensive Python virtual environment setup guide in CLAUDE.md
3. **Troubleshooting Guide**: Added common error scenarios and solutions for future sessions

### Current System Status
- **All Core Features**: ✅ Working (negative pricing, duck curves, rolling analysis)
- **Data Pipeline**: ✅ Stable (includes 2025 forecast data)
- **Chart Generation**: ✅ Enhanced (clear threshold labeling)
- **Python Environment**: ✅ Documented (comprehensive venv guide)

## ✅ COMPLETED: CLI Output Flag Improvements & Negative Pricing Timechart 📊⚡

### Latest Session Accomplishments
Successfully implemented negative pricing timechart feature and cleaned up CLI output handling:

#### 🎯 New Negative Pricing Timechart Feature
1. **Added Chart Type Option**: Extended `--chart-type` parameter with `timechart` option showing daily hours with negative/near-zero prices
2. **Daily Hours Calculation**: New `calculate_daily_hours_timeseries()` function aggregating hourly data into daily counts
3. **Terminal & PNG Visualization**: Both terminal and high-resolution PNG charts showing time series of negative pricing hours
4. **Smart Date Handling**: Auto-adjusting x-axis labels based on data range (daily/weekly/monthly)
5. **Summary Statistics**: Charts include average hours, max hours, and days with negative prices

#### 🔧 CLI Output Flag Improvements  
1. **Clean --png Flag**: Added `--png` flag for auto-generated filenames (no argument needed)
2. **Removed Clunky "auto"**: Eliminated confusing `-o auto` syntax and `__AUTO_GENERATE__` logic
3. **Fixed Duplicate Parameters**: Resolved CLI warnings about duplicate `--output` parameters
4. **Consistent Behavior**: All commands (`chart`, `duck-factor`, `negative-pricing`) now have unified output handling

#### 📈 Usage Examples
```bash
# Terminal timechart (default)
angry-pixie negative-pricing --region DE --start-date 2024 --chart-type timechart

# Auto-generated PNG filename
angry-pixie negative-pricing --region DE --start-date 2024 --chart-type timechart --png

# Custom PNG filename  
angry-pixie negative-pricing --region DE --start-date 2024 --chart-type timechart -o my_chart.png
```

#### 🎨 Timechart Insights
The new timechart reveals powerful seasonal patterns:
- **Summer peaks**: May-August show 5-15+ hours of negative pricing per day
- **Winter lows**: December-February show minimal negative pricing
- **Daily variations**: Some days have 0 hours, others up to 16+ hours
- **Trend visibility**: Multi-year data shows renewable energy adoption progress

### Current System Status
- **All Core Features**: ✅ Working (negative pricing analysis, duck curves, rolling analysis, timecharts)
- **Data Pipeline**: ✅ Stable (includes 2025 forecast data)
- **Chart Generation**: ✅ Enhanced (terminal & PNG with clean CLI)
- **Code Quality**: ✅ Improved (removed duplicate parameters, simplified logic)

## ✅ COMPLETED: Weekly & Monthly Aggregation for Negative Pricing Timecharts 📊📅

### Latest Session Accomplishments
Successfully implemented comprehensive aggregation support for negative pricing timechart analysis:

#### 🎯 New Aggregation Level Feature
1. **Added --aggregation-level Parameter**: Extended CLI with daily/weekly/monthly options for timechart mode
2. **Weekly Aggregation**: New `calculate_weekly_hours_timeseries()` function using pandas weekly periods
3. **Monthly Aggregation**: New `calculate_monthly_hours_timeseries()` function using pandas monthly periods
4. **Unified Interface**: New `calculate_aggregated_hours_timeseries()` wrapper function for all aggregation levels
5. **Smart Filename Generation**: Auto-generated PNG filenames include aggregation level (e.g., `timechart-weekly`)

#### 🔧 Enhanced Chart Functions
1. **Terminal Charts**: Updated `create_terminal_negative_pricing_timechart()` to support all aggregation levels
2. **PNG Charts**: Updated `create_png_negative_pricing_timechart()` with intelligent x-axis formatting for different time periods
3. **Dynamic Labeling**: Chart titles, y-axis labels, and summaries adapt to aggregation level
4. **Smart X-Axis**: Automatic formatting (YYYY-MM-DD for daily, YYYY-Wxx for weekly, YYYY-MM for monthly)

#### 📈 Enhanced Usage Examples
```bash
# Daily timechart (default)
angry-pixie negative-pricing --region DE --start-date 2024 --chart-type timechart

# Weekly aggregation 
angry-pixie negative-pricing --region DE --start-date 2024-05 --chart-type timechart --aggregation-level weekly

# Monthly aggregation with PNG output
angry-pixie negative-pricing --region AT --start-date 2024 --chart-type timechart --aggregation-level monthly --png

# Creates: images/negative-pricing-timechart-monthly_at_20240101_20250613.png
```

#### 🎨 Aggregation Benefits
The new aggregation levels provide different analytical perspectives with comparable scales:
- **Daily**: Fine-grained view of day-to-day variations, seasonal trends (0-24 hours)
- **Weekly**: Average daily hours within weeks, smoothed medium-term trends (0-24 hours avg)
- **Monthly**: Average daily hours within months, long-term seasonal patterns (0-24 hours avg)
- **Multi-year Analysis**: Clear visualization of renewable energy adoption progress

**Key Fix**: Weekly and monthly aggregations show average daily hours within each period, making all aggregation levels directly comparable on the same 0-24 hour scale.

### Updated System Status
- **All Core Features**: ✅ Working (negative pricing analysis, duck curves, rolling analysis, timecharts with aggregation)
- **Data Pipeline**: ✅ Stable (includes 2025 forecast data) 
- **Chart Generation**: ✅ Enhanced (terminal & PNG with daily/weekly/monthly aggregation)
- **CLI Interface**: ✅ Comprehensive (clean parameter handling, auto-filenames, aggregation options)
- **Code Quality**: ✅ Improved (modular aggregation functions, consistent naming)

## Session Handoff Notes

### For Next Session
1. **Environment Setup**: Always start with `source venv/bin/activate` (documented in CLAUDE.md)
2. **Ready for Enhancement**: Core system is stable, ready for new features or model improvements
3. **High Priority Next Tasks**:
   - ✅ ~~**Negative Pricing Aggregation Levels**: Add weekly and monthly aggregation options for timecharts~~ **COMPLETED**
   - **Solar Model Improvements**: Enhance solar potential accuracy by considering wind/hydro integration
   - **Cross-Regional Analysis**: Implement comparative analysis features across multiple countries
   - **Advanced Time Series**: Add year-over-year comparison overlays for timecharts

### Key Files to Know
- `src/angry_pixie_pricing/data/base.py` - Abstract caching and data source framework
- `src/angry_pixie_pricing/data/energy_charts.py` - EnergyCharts.info API implementation  
- `src/angry_pixie_pricing/charts/terminal.py` - All visualization functions (includes enhanced timechart functions with aggregation)
- `src/angry_pixie_pricing/main.py` - CLI command definitions (updated with --png flag and aggregation support)
- `src/angry_pixie_pricing/analysis/negative_pricing.py` - Includes aggregation functions (`calculate_aggregated_hours_timeseries()`)
- `src/angry_pixie_pricing/utils/cli_options.py` - CLI option definitions (includes aggregation-level parameter)

### Development Workflow
1. Use TodoWrite tool to track session progress
2. Always activate venv: `source venv/bin/activate`
3. Commit frequently with descriptive messages
4. Update this PLAN.md file when major features complete
5. Run `black src/` before committing for consistent formatting

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

*Last updated: Weekly & monthly aggregation for negative pricing timecharts session*
*Current status: All major features working, comprehensive timechart aggregation (daily/weekly/monthly), enhanced CLI*
*Next priorities: Solar model improvements, cross-regional analysis, advanced time series features*