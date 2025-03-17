from dataclasses import dataclass, field
from typing import Callable, Any, Union
from enum import Enum
import numpy as np

class Season(Enum):
    """Standard meteorological seasons."""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"  # or FALL
    
    # Special periods
    ANNUAL = "annual"
    GROWING_SEASON = "growing_season"
    
    def __str__(self):
        return self.value

@dataclass
class SeasonDefinition:
    """Definition of a season with potentially different months by hemisphere."""
    name: str
    nh_months: list[int]  # Northern Hemisphere months (1=Jan, 12=Dec)
    sh_months: list[int]  # Southern Hemisphere months
    spans_year_boundary: bool = False  # Does the season cross the Dec-Jan boundary?
    
    def get_months_for_hemisphere(self, latitude: float | np.ndarray) -> list[int]:
        """Get the months for a given latitude/hemisphere."""
        if isinstance(latitude, np.ndarray):
            # For gridded data, return a dictionary mapping months to boolean masks
            nh_mask = latitude >= 0
            sh_mask = latitude < 0
            return {
                "nh_mask": nh_mask,
                "sh_mask": sh_mask,
                "nh_months": self.nh_months,
                "sh_months": self.sh_months
            }
        else:
            # For a single latitude value
            return self.nh_months if latitude >= 0 else self.sh_months

@dataclass
class TemporalWindow:
    """Definition of a temporal window for metric calculation."""
    season: SeasonDefinition | None = None  # Season (if seasonal metric)
    years: int = 30  # Number of years for the window (e.g., 30 for climate normals)
    sub_period: tuple[int, int] | None = None  # Sub-period within a year (day of year start, end)
    
    def is_seasonal(self) -> bool:
        """Check if this is a seasonal window."""
        return self.season is not None and self.season.name != str(Season.ANNUAL)

@dataclass
class MetricDefinition:
    """Defines a climate metric for the atlas."""
    name: str
    description: str
    variables: list[str]  # List of CMIP6 variables needed for metric calculation
    temporal_window: TemporalWindow  # Temporal aggregation window
    calculation_func: Callable  # Function to compute metric
    output_units: str
    threshold_based: bool = False  # Is this a threshold-based metric (e.g., frost days)
    threshold: float | None = None  # Threshold value if applicable
    spatial_operation: str = "mean"  # Operation for spatial aggregation (mean, max, min, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

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