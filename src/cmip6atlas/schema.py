"""Object containing data organization of CMIP6 model outputs on NEX bucket."""

GDDP_CMIP6_SCHEMA: dict[str, str | list[str] | int] = {
    "bucket": "nex-gddp-cmip6",
    "prefix": "NEX-GDDP-CMIP6",
    "variables": [
        "hurs",
        "huss",
        "pr",
        "rlds",
        "rsds",
        "sfcWind",
        "tas",
        "tasmax",
        "tasmin",
    ],
    "scenarios": ["historical", "ssp126", "ssp245", "ssp370", "ssp585"],
    "min_year": 1950,
    "max_year": 2100,
    "historical_end_year": 2014,
}
