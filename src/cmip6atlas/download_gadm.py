import pandas as pd
import requests
import importlib.resources
import os
import time
import argparse
import sys

DEFAULT_CSV_FILENAME = "gadm_countries.csv"
DEFAULT_DOWNLOAD_DIR = "country-bounds"
DEFAULT_REQUEST_DELAY = 0.5  # seconds
DEFAULT_TIMEOUT = 60  # seconds for requests

# GADM download URL template, hardcoded to v4.1
# Note: GADM URLs can change. Verify this base if downloads fail.
URL_TEMPLATE = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{}_{}.json"


def get_script_directory():
    """Gets the directory where the currently running script resides."""
    return os.path.dirname(os.path.abspath(__file__))


def get_default_csv_path():
    """Returns the path to the default CSV file within the package data."""
    try:
        resource_path = importlib.resources.files("cmip6atlas").joinpath(
            DEFAULT_CSV_FILENAME
        )
        # Ensure the path exists before returning it as a string
        if resource_path.is_file():
            return str(resource_path)
        else:
            # This should ideally not happen if packaging is correct
            print(
                f"Error: Default CSV '{DEFAULT_CSV_FILENAME}' not found within package 'cmip6atlas'."
            )
            return None
    except ModuleNotFoundError:
        # Handle case where the package itself isn't found
        print(f"Error: Package 'cmip6atlas' not found.")
        return None
    except Exception as e:
        print(f"Error accessing package resource '{DEFAULT_CSV_FILENAME}': {e}")
        return None


def read_config_csv(csv_path):
    """
    Reads the configuration CSV file.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The DataFrame containing country config.
        None: If there's an error reading the file.
    """
    try:
        df = pd.read_csv(csv_path)
        # Basic validation: Check if required columns exist
        required_cols = ["country_code", "admin_level"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(
                f"Error: Input CSV '{csv_path}' is missing required columns: {missing}"
            )
            return None
        print(f"Successfully read configuration from: {csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{csv_path}'")
        return None
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return None


def download_single_file(
    url, destination, skip_existing_flag=False, timeout=DEFAULT_TIMEOUT
):
    """
    Downloads a single file from a URL to a destination.

    Args:
        url (str): The URL to download from.
        destination (str): The local path to save the file.
        skip_existing_flag (bool): If True, skip download if file exists.
        timeout (int): Request timeout in seconds.

    Returns:
        tuple: (bool: success, str: status_message)
               Status messages: "downloaded", "skipped_existing", "http_error",
                                "request_error", "other_error"
    """
    if skip_existing_flag and os.path.exists(destination):
        print(f"Skipping existing file: {os.path.basename(destination)}")
        return True, "skipped_existing"

    print(f"Attempting download: {os.path.basename(destination)} from {url}")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        # Ensure directory exists before writing (though created in main func)
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded: {os.path.basename(destination)}")
        return True, "downloaded"
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error downloading {url}: {e}")
        if e.response.status_code == 404:
            print(
                f"  -> Hint: 404 Not Found often means the specific country code/admin level combination ({os.path.basename(destination)}) does not exist on the server."
            )
        return False, "http_error"
    except requests.exceptions.RequestException as e:
        print(f"Request Error downloading {url}: {e}")
        return False, "request_error"
    except Exception as e:
        print(f"Error processing {url} to {destination}: {e}")
        return False, "other_error"


def process_downloads(config_df, download_dir, skip_existing, request_delay):
    """
    Iterates through the config DataFrame and downloads the specified files.

    Args:
        config_df (pd.DataFrame): DataFrame with download configuration.
        download_dir (str): Directory to save downloaded files.
        skip_existing (bool): Whether to skip existing files.
        request_delay (float): Delay between download requests.

    Returns:
        dict: A summary of the download process containing counts.
    """
    total_rows = len(config_df)
    summary = {
        "total_entries": total_rows,
        "skipped_invalid": 0,
        "downloaded": 0,
        "skipped_existing": 0,
        "errors": 0,
    }

    print(f"\nStarting download process for {total_rows} entries...")

    for idx, row in config_df.iterrows():
        country_name = row.get("country_name", "N/A")  # Optional but nice for logging
        country_code = row.get("country_code")
        admin_level = row.get("admin_level")

        # --- Input Validation for the row ---
        if (
            pd.isna(country_code)
            or not isinstance(country_code, str)
            or len(country_code.strip()) != 3
        ):
            print(
                f"Skipping row {idx+1} ({country_name}): Invalid or missing country_code '{country_code}'"
            )
            summary["skipped_invalid"] += 1
            continue

        country_code_upper = country_code.strip().upper()

        if pd.isna(admin_level):
            print(f"Skipping row {idx+1} ({country_name}): Missing admin_level")
            summary["skipped_invalid"] += 1
            continue
        try:
            admin_level_int = int(admin_level)
        except (ValueError, TypeError):
            print(
                f"Skipping row {idx+1} ({country_name}): Invalid admin_level '{admin_level}' (must be an integer)"
            )
            summary["skipped_invalid"] += 1
            continue

        url = URL_TEMPLATE.format(country_code_upper, admin_level_int)
        filename = f"gadm41_{country_code_upper}_{admin_level_int}.json"
        destination = os.path.join(download_dir, filename)

        success, status = download_single_file(
            url, destination, skip_existing_flag=skip_existing
        )

        if success:
            if status == "downloaded":
                summary["downloaded"] += 1
            elif status == "skipped_existing":
                summary["skipped_existing"] += 1
        else:
            summary["errors"] += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == total_rows:
            print(f"--- Processed {idx+1}/{total_rows} entries ---")

        time.sleep(request_delay)

    return summary


def download_gadm_data(
    csv_path=None,
    download_dir=DEFAULT_DOWNLOAD_DIR,
    skip_existing=False,
    request_delay=DEFAULT_REQUEST_DELAY,
):
    """
    Downloads GADM GeoJSON files based on a configuration CSV.

    Args:
        csv_path (str, optional): Path to the configuration CSV file.
                                  Defaults to 'gadm_countries.csv' in the script's directory.
        download_dir (str, optional): Directory to store downloaded files.
                                      Defaults to 'gadm_data'.
        skip_existing (bool, optional): If True, skip downloading files that already exist.
                                        Defaults to False.
        request_delay (float, optional): Delay in seconds between download requests.
                                         Defaults to 0.5.

    Returns:
        bool: True if the process started successfully (CSV read, dir created),
              False otherwise. The final summary is printed to stdout.
    """
    print("--- Starting GADM Data Download ---")

    # Determine CSV path
    if csv_path is None:
        csv_path = get_default_csv_path()
        print(f"No CSV path provided, using default: {csv_path}")

    # Create the download directory
    try:
        os.makedirs(download_dir, exist_ok=True)
        print(f"Using download directory: {os.path.abspath(download_dir)}")
    except OSError as e:
        print(f"Error creating directory {download_dir}: {e}")
        return False  # Cannot proceed without download directory

    # Read the configuration
    config_df = read_config_csv(csv_path)
    if config_df is None:
        return False  # Error message already printed by read_config_csv

    # Process the downloads
    summary = process_downloads(config_df, download_dir, skip_existing, request_delay)

    # Print Final Summary
    print("\n----- Download process completed -----")
    print(f"Total entries in CSV:          {summary['total_entries']}")
    print(f"Entries skipped (invalid data): {summary['skipped_invalid']}")
    print(f"Files successfully downloaded:   {summary['downloaded']}")
    print(f"Files skipped (already existed): {summary['skipped_existing']}")
    print(f"Download errors encountered:     {summary['errors']}")
    print("------------------------------------")

    return True


def parse_arguments():
    """Parses command-line arguments when the script is run directly."""
    parser = argparse.ArgumentParser(
        description="Download GADM GeoJSON administrative boundary files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Shows defaults in help
    )
    # CSV is now optional, defaults handled by main function
    parser.add_argument(
        "-c",
        "--csv",
        dest="csv_file",
        default=None,  # Let main function handle default path logic
        help=f"Path to the input CSV file (defaults to {DEFAULT_CSV_FILENAME} next to the script).",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory to store downloaded GeoJSON files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If set, skip downloading files that already exist in the destination directory.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="Delay in seconds between download requests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    success = download_gadm_data(
        csv_path=args.csv_file,
        download_dir=args.directory,
        skip_existing=args.skip_existing,
        request_delay=args.delay,
    )

    if not success:
        sys.exit(1)
