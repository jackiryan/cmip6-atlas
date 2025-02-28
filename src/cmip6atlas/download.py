"""
Copyright (c) 2025 Jacqueline Ryan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import argparse
import concurrent.futures
from dataclasses import dataclass
from typing import Any, cast
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from mypy_boto3_s3.client import S3Client


GDDP_CMIP6_SCHEMA = {
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
    "realization": "r1i1p1f1",
}


@dataclass(frozen=True)
class GranuleSubset:
    start_year: int
    end_year: int
    variable: str
    scenario: str
    models: list[str]


class DownloadError(Exception):
    pass


def create_s3_client() -> S3Client:
    """Create an S3 client with unsigned config for accessing public NEX bucket."""
    return cast(S3Client, boto3.client("s3", config=Config(signature_version=UNSIGNED)))


def get_available_models(s3_client: S3Client, bucket: str, prefix: str) -> list[str]:
    """Get the list of available climate models in the bucket."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")

    models = []
    for found_dir in response.get("CommonPrefixes", []):
        dir_str = found_dir.get("Prefix")
        if dir_str is not None:
            model_name = dir_str.split("/")[-2]
            models.append(model_name)

    if models == []:
        raise DownloadError(
            f"could not retrieve available models from address: s3://{bucket}/{prefix}"
        )

    return models


def get_available_files(
    s3_client: S3Client,
    schema: dict[str, Any],
    model: str,
    scenario: str,
    variable: str,
    start_year: int,
    end_year: int,
) -> list[tuple[str, str, int]]:
    """
    Get the list of available files for the given parameters.

    Returns a list of tuples containing (s3_key, filename, file_size)
    """
    files: list[tuple[str, str, int]] = []

    # Handle time ranges than extend between historical and projected time ranges,
    # e.g., 1991-2020 for normals calculations.
    hist_end_year = int(schema["historical_end_year"])
    if start_year < hist_end_year and scenario != "historical":
        if end_year < hist_end_year:
            intmd_end_year = end_year
        else:
            intmd_end_year = hist_end_year
        # recursive function call to get historical files
        files.extend(
            get_available_files(
                s3_client,
                schema,
                model,
                "historical",
                variable,
                start_year,
                intmd_end_year,
            )
        )
        start_year = hist_end_year + 1
    prefix_str = str(schema["prefix"]).rstrip("/")

    prefix = (
        f"{schema['prefix']}/{model}/{scenario}/{schema['realization']}/{variable}/"
    )

    response = s3_client.list_objects_v2(Bucket=schema["bucket"], Prefix=prefix)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = os.path.basename(key)

        # Extract year from filename
        # Example filename: tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2025.nc
        # Will throw ValueError or IndexError if filename does not conform to convention
        try:
            file_noext = os.path.splitext(filename)[0]
            fname_parts = file_noext.split("_")

            # if v is in the part of the filename where a year is expected,
            # it is a version number
            # example: tas_day_TaiESM1_ssp585_r1i1p1f1_gn_2015_v1.1.nc
            if "v" in fname_parts[-1]:
                year_part = fname_parts[-2]
            else:
                year_part = fname_parts[-1]

            file_year = int(year_part)
            if start_year <= file_year <= end_year:
                # versions are sorted in the bucket, so we can pop the previous
                # version if we encounter a higher one
                if "v" in fname_parts[-1]:
                    files.pop()
                files.append((key, filename, obj["Size"]))
        except (ValueError, IndexError):
            print(f"Unable to parse year from filename: {filename}")
            continue

    return files


def download_file(s3_client: S3Client, bucket: str, key: str, local_path: str) -> bool:
    """
    Download a file from S3 to the local path.
    Returns True if downloaded, False if skipped.
    """
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        response = s3_client.head_object(Bucket=bucket, Key=key)
        if file_size == response["ContentLength"]:
            print(f"Skipping {os.path.basename(local_path)} (already exists)")
            return False

    directory = os.path.dirname(local_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Downloading {os.path.basename(local_path)}...")
    s3_client.download_file(bucket, key, local_path)
    return True


def download_files_parallel(
    s3_client: S3Client,
    bucket: str,
    files: list[tuple[str, str, int]],
    output_dir: str,
    max_workers: int = 5,
) -> int:
    """
    Download files in parallel using ThreadPoolExecutor.
    Returns the count of files actually downloaded.
    """
    download_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for key, filename, _ in files:
            local_path = os.path.join(output_dir, filename)
            futures.append(
                executor.submit(download_file, s3_client, bucket, key, local_path)
            )

        for future in concurrent.futures.as_completed(futures):
            if future.result():
                download_count += 1

    return download_count


def format_size(size_bytes: int) -> str:
    """Format file size in a human-readable format."""
    size_float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_float < 1024 or unit == "TB":
            return f"{size_float:.2f} {unit}"
        size_float /= 1024
    # Default return statement for type checking, should never occur
    return f"{size_float:.2f} B"


def validate_inputs(schema: dict[str, Any], inputs: GranuleSubset) -> None:
    errmsg: list[str] = []

    # Validate years
    if not (schema["min_year"] <= inputs.start_year <= schema["max_year"]):
        errmsg.append(
            f"Start year: {inputs.start_year} must be between {schema['min_year']} "
            f"and {schema['max_year']}"
        )

    if not (schema["min_year"] <= inputs.end_year <= schema["max_year"]):
        errmsg.append(
            f"End year: {inputs.end_year} must be between {schema['min_year']} and "
            f"{schema['max_year']}"
        )

    if inputs.start_year > inputs.end_year:
        errmsg.append(
            f"Start year cannot be after end year, {inputs.start_year} > "
            f"{inputs.end_year}"
        )

    # Validate variable (handled by argparse if using CLI)
    if inputs.variable not in schema["variables"]:
        errmsg.append(
            f"{inputs.variable} is not available, choices are: "
            f"{', '.join(schema['variables'])}"
        )

    # Validate scenario (handled by argparse if using CLI)
    if inputs.scenario not in schema["scenarios"]:
        errmsg.append(
            f"{inputs.scenario} is not available, choices are: "
            f"{', '.join(schema['scenarios'])}"
        )

    # Validate query is historical data year if using historical scenario. The inverse
    # case (historical year with non-historical scenario) will trigger
    # get_available_files to use the historical data path automatically.
    years_in_range = set(range(inputs.start_year, inputs.end_year + 1))
    historical_years = set(range(
        schema["min_year"],
        schema["historical_end_year"] + 1
    ))
    if inputs.scenario == "historical" and not years_in_range.issubset(
        historical_years
    ):
        errmsg.append(
            f"specified historical data, but query range {inputs.start_year}-"
            f"{inputs.end_year} extends outside historical record ending in "
            f"{schema['historical_end_year']}"
        )

    if errmsg != []:
        raise ValueError("; ".join(errmsg))


def download_granules(
    variable: str,
    scenario: str,
    start_year: int,
    end_year: int | None = None,
    output_dir: str = "./nex-gddp-data",
    models: list[str] | None = None,
    exclude_models: list[str] | None = None,
    max_workers: int = 5,
    skip_prompt: bool = True,
    schema: dict[str, Any] | None = None,
) -> None:
    """
    Download a set of granules (netCDF files) from the NEX-GDDP-CMIP6 bucket, with
    default behavior to pull all available models for a given variable/scenario
    combination.

    Usage:
        ```
        from cmip6atlas.download import download_granules
        # Download all available model outputs for the surface temperature (tas)
        # variable under the SSP5-8.5 scenario, with an interactive prompt.
        download_granules(
            "tas",
            "ssp585",
            2025,
            skip_prompt=False,
        )
        ```
    Args:
        variable (str): A string describing the CMIP6 variable to download.
        start_year (int): Integer starting year of the data query.
        end_year (int | None): Optional end year, default is the start year.
        scenario (str): Emissions scenario (SSP) for the data query. The "historical"
            scenario is only available for queries between 1950-2014.
        output_dir (str): String path to store the downloaded files. Will be created
            if it does not yet exist.
        models (list[str] | None): Optionally provide a list of climate models to use
            in the query. Unavailable models will be ignored, invalid model names will
            throw a ValueError.
        exclude_models (list[str] | None): Optionally provide a list of models to
            exclude from the query. Should not be used in combination with the
            models keyword.
        max_workers (int): Number of parallel threads to use for downloading files.
        skip_prompt (bool): If True, skip the input prompt asking for confirmation of
            the amount of data to download. Default is True.
        schema (dict[str, Any] | None): Optionally specify a dataset schema for
            querying a different s3 bucket. See GDDP_CMIP6_SCHEMA for expected format.

    Returns:
        None

    Raises:
        ValueError: If any of the query input arguments are invalid.
        DownloadError: If an S3 query fails.
    """
    # Leave option to download CMIP5, CMIP7 or other similarly organized datasets from
    # NEX later. For now GDDP_CMIP6_SCHEMA is the only supported dataset
    if not schema:
        schema = GDDP_CMIP6_SCHEMA

    s3_client = create_s3_client()

    # Determine the available models for this dataset schema
    bucket = str(schema["bucket"])
    prefix_str = str(schema["prefix"])
    if not prefix_str.endswith("/"):
        prefix_str += "/"
    available_models = get_available_models(s3_client, bucket, prefix_str)

    if not end_year:
        end_year = start_year

    # For now, this relies on input to be exact string match
    if models and not exclude_models:
        use_models = list(set(models) & set(available_models))
    elif exclude_models and not models:
        use_models = list(set(available_models) - set(exclude_models))
    elif not models and not exclude_models:
        use_models = available_models
    else:
        raise ValueError("cannot exclude and include models in same query.")

    inputs = GranuleSubset(
        start_year,
        end_year,
        variable.lower(),
        scenario.lower(),
        use_models
    )
    # Will throw ValueError on validation failure
    validate_inputs(schema, inputs)

    files_to_download = []
    total_size = 0
    existing_files_size = 0
    print("Searching for granules matching criteria...")
    for model in inputs.models:
        model_files = get_available_files(
            s3_client,
            schema,
            model,
            inputs.scenario,
            inputs.variable,
            inputs.start_year,
            inputs.end_year,
        )

        for key, filename, size in model_files:
            local_path = os.path.join(output_dir, filename)
            if os.path.exists(local_path) and os.path.getsize(local_path) == size:
                existing_files_size += size
            else:
                total_size += size
                files_to_download.append((key, filename, size))

    if not files_to_download and existing_files_size == 0:
        print("No files found matching your criteria. Please check your parameters.")
        return

    if not skip_prompt and files_to_download:
        print(
            f"After this operation, {len(files_to_download)} granules totaling "
            f"{format_size(total_size)} will be added to {output_dir}"
        )
        confirmation = input("\nProceed with download? [y/N] ").lower()
        if confirmation != "y":
            print("Download canceled.")
            return
    elif files_to_download == []:
        print("\nAll files already exist locally. No downloads needed.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDownloading {len(files_to_download)} files...")
    download_count = download_files_parallel(
        s3_client, bucket, files_to_download, output_dir, max_workers
    )
    print(f"\nDownload complete. {download_count} files downloaded.")


def cli() -> None:
    """
    Command Line Interface for downloading files independently of
    other processing steps.
    """
    parser = argparse.ArgumentParser(
        description="Download climate model data from NEX-GDDP-CMIP6 S3 bucket"
    )

    schema = GDDP_CMIP6_SCHEMA

    variables = cast(list[str], schema["variables"])
    scenarios = cast(list[str], schema["scenarios"])
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        choices=variables,
        help="Climate variable to download",
    )

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
        "--scenario",
        type=str,
        required=True,
        choices=scenarios,
        help="Climate scenario (historical is only valid for 1950-2014)",
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Specific models to download (default: all available models)",
    )

    parser.add_argument(
        "--exclude-models",
        type=str,
        nargs="+",
        help="Exclude specific models from the query",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./nex-gddp-data",
        help="Directory to save downloaded files",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel downloads",
    )

    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # schema option is not provided in cli due to only one option (for now)
    download_granules(
        args.variable,
        args.scenario,
        args.start_year,
        args.end_year,
        args.output_dir,
        args.models,
        args.exclude_models,
        args.max_workers,
        args.yes,
    )
