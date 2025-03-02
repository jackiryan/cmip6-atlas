import argparse
from typing import cast

from cmip6atlas.schema import GDDP_CMIP6_SCHEMA


def get_parser(description: str, task: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    schema = GDDP_CMIP6_SCHEMA
    variables = cast(list[str], schema["variables"])
    scenarios = cast(list[str], schema["scenarios"])

    # Base parser parameters, common to all tasks
    parser.add_argument(
        "-v",
        "--variable",
        type=str,
        required=True,
        choices=variables,
        help="Climate variable to download",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        required=True,
        choices=scenarios,
        help=f"Climate scenario (historical is only valid for {schema['min_year']}-{schema['historical_end_year']})",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific models to use (default: all available models)",
    )

    parser.add_argument(
        "--exclude-models",
        type=str,
        nargs="+",
        help="Exclude specific models from processing",
    )

    if task == "download":
        dft_output_dir = "./nex-gddp-cmip6"
        parser.add_argument(
            "--yes", action="store_true", help="Skip confirmation prompt"
        )
    else:
        dft_output_dir = "./ensemble-means"
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=dft_output_dir,
        help="Directory to save processed files",
    )

    if task in ["download", "window"]:
        parser.add_argument(
            "--start-year",
            type=int,
            required=True,
            help=f"Start year (between {schema['min_year']} and {schema['max_year']})",
        )

        parser.add_argument(
            "--end-year",
            type=int,
            help=(
                f"End year (between {schema['min_year']} and {schema['max_year']}, "
                "defaults to start-year)"
            ),
        )
        parser.add_argument(
            "--max-workers",
            type=int,
            default=5,
            help="Maximum number of parallel workers",
        )

    if task in ["ensemble", "window"]:
        parser.add_argument(
            "-i",
            "--input-dir",
            default="./nex-gddp-cmip6",
            help="Directory with individual climate model NetCDF files",
        )

    if task == "ensemble":
        parser.add_argument("--year", type=int, help="Year to process")
        parser.add_argument(
            "-m", "--month", type=int, help="Month to subset model data"
        )

    return parser
