from cmip6atlas.metrics.definitions import *
from cmip6atlas.metrics.processes import *

# Standard season definitions
SEASON_DEFINITIONS = {
    Season.WINTER: SeasonDefinition(
        name=str(Season.WINTER),
        nh_months=[12, 1, 2],  # DJF in Northern Hemisphere
        sh_months=[6, 7, 8],   # JJA in Southern Hemisphere
        spans_year_boundary=True  # December is from previous year
    ),
    Season.SPRING: SeasonDefinition(
        name=str(Season.SPRING),
        nh_months=[3, 4, 5],   # MAM in Northern Hemisphere
        sh_months=[9, 10, 11]  # SON in Southern Hemisphere
    ),
    Season.SUMMER: SeasonDefinition(
        name=str(Season.SUMMER),
        nh_months=[6, 7, 8],   # JJA in Northern Hemisphere
        sh_months=[12, 1, 2],  # DJF in Southern Hemisphere
        spans_year_boundary=True  # December is from previous year for SH
    ),
    Season.AUTUMN: SeasonDefinition(
        name=str(Season.AUTUMN),
        nh_months=[9, 10, 11],  # SON in Northern Hemisphere
        sh_months=[3, 4, 5]     # MAM in Southern Hemisphere
    ),
    Season.ANNUAL: SeasonDefinition(
        name=str(Season.ANNUAL),
        nh_months=list(range(1, 13)),  # All months
        sh_months=list(range(1, 13))   # All months
    ),
    # Custom growing season example
    Season.GROWING_SEASON: SeasonDefinition(
        name=str(Season.GROWING_SEASON),
        nh_months=[4, 5, 6, 7, 8, 9],  # Apr-Sep in Northern Hemisphere
        sh_months=[10, 11, 12, 1, 2, 3],  # Oct-Mar in Southern Hemisphere
        spans_year_boundary=True  # SH growing season spans year boundary
    )
}

# Example metrics registry with seasonal metrics
CLIMATE_METRICS = {
    "annual_temp_normal": MetricDefinition(
        name="annual_temp_normal",
        description="Annual Temperature normal (30-year average)",
        variables=["tas"],
        temporal_window=TemporalWindow(
            season=SEASON_DEFINITIONS[Season.ANNUAL],
            years=5
        ),
        calculation_func=lambda datasets, **kwargs: calculate_seasonal_temp_normal(
            datasets,
            **kwargs
        ),
        output_units="°C",
        metadata={"long_name": "Annual Temperature Normal"}
    ),
    '''
    "annual_precip_normal": MetricDefinition(
        name="annual_precip_normal",
        description="Annual precipitation normal (30-year average)",
        variables=["pr"],
        temporal_window=TemporalWindow(
            season=SEASON_DEFINITIONS[Season.ANNUAL],
            years=30
        ),
        calculation_func=lambda datasets, **kwargs: calculate_precip_normal(datasets, **kwargs),
        output_units="mm/year",
        metadata={"long_name": "Annual Precipitation Normal"}
    ),
    '''
    "winter_mean_temp": MetricDefinition(
        name="winter_mean_temp",
        description="Winter (DJF in NH, JJA in SH) mean temperature normal",
        variables=["tas"],
        temporal_window=TemporalWindow(
            season=SEASON_DEFINITIONS[Season.WINTER],
            years=30
        ),
        calculation_func=lambda datasets, **kwargs: calculate_seasonal_temp_normal(
            datasets, 
            season=SEASON_DEFINITIONS[Season.WINTER],
            **kwargs
        ),
        output_units="°C",
        metadata={"long_name": "Winter Mean Temperature Normal"}
    ),
    "growing_season_length": MetricDefinition(
        name="growing_season_length",
        description="Average length of growing season (days above 5°C)",
        variables=["tas"],
        temporal_window=TemporalWindow(years=30),
        calculation_func=lambda datasets, **kwargs: calculate_growing_season_length(
            datasets, 
            base_temp=5.0,
            **kwargs
        ),
        output_units="days",
        threshold_based=True,
        threshold=5.0,
        metadata={"long_name": "Growing Season Length", "threshold": "5.0 °C"}
    ),
    "extreme_heat_days": MetricDefinition(
        name="extreme_heat_days",
        description="Annual average number of days with max temperature above 35°C",
        variables=["tasmax"],
        temporal_window=TemporalWindow(
            season=SEASON_DEFINITIONS[Season.ANNUAL],
            years=30
        ),
        calculation_func=lambda datasets, **kwargs: calculate_threshold_days(
            datasets, 
            threshold=35.0, 
            operation="gt",
            **kwargs
        ),
        output_units="days/year",
        threshold_based=True,
        threshold=35.0,
        metadata={"long_name": "Extreme Heat Days", "threshold": "35.0 °C"}
    )
}