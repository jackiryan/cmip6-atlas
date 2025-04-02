from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from typing import Callable, Any

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
        return self.season is not None #and self.season.name != str(Season.ANNUAL)

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