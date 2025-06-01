#!/usr/bin/env python3
"""
Create a global regions file with unique IDs from GADM administrative boundaries.

This script:
1. Downloads GADM data for each country/territory listed in gadm_countries.csv
2. Combines all individual country files into a single global regions dataset
3. Assigns unique region IDs to each administrative feature
4. Saves the result as a GeoJSON file

Usage:
    python create_global_regions.py [options]
"""

import argparse
import importlib.resources
import json
import os
import sys
import geopandas as gpd  # type: ignore [import-untyped]
import pandas as pd

from .download_gadm import download_gadm_data


def get_default_csv_path() -> str | None:
    """Returns the path to the default CSV file within the package data."""
    try:
        resource_path = importlib.resources.files("cmip6atlas").joinpath(
            "gadm_countries.csv"
        )
        if resource_path.is_file():
            return str(resource_path)
        else:
            print("Error: Default CSV 'gadm_countries.csv' not found within package.")
            return None
    except Exception as e:
        print(f"Error accessing package resource 'gadm_countries.csv': {e}")
        return None


def read_countries_config(csv_path: str) -> pd.DataFrame | None:
    """Read the countries configuration CSV file."""
    try:
        df = pd.read_csv(csv_path)
        required_cols = ["country_name", "country_code", "admin_level"]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: CSV missing required columns: {missing}")
            return None
        print(f"Read configuration for {len(df)} countries/territories")
        return df
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        return None


def load_gadm_files(
    countries_df: pd.DataFrame, gadm_dir: str
) -> list[gpd.GeoDataFrame]:
    """Load all GADM GeoJSON files into a list of GeoDataFrames."""
    gdfs = []
    missing_files = []

    print(f"Loading GADM files from {gadm_dir}...")

    for _, row in countries_df.iterrows():
        country_code = row["country_code"].upper()
        admin_level = int(row["admin_level"])
        country_name = row["country_name"]

        filename = f"gadm41_{country_code}_{admin_level}.json"
        filepath = os.path.join(gadm_dir, filename)

        if not os.path.exists(filepath):
            missing_files.append((country_name, filename))
            continue

        try:
            gdf = gpd.read_file(filepath)

            # Add metadata columns
            gdf["source_country_code"] = country_code
            gdf["source_country_name"] = country_name
            gdf["source_admin_level"] = admin_level
            gdf["source_filename"] = filename

            gdfs.append(gdf)
            print(f"  Loaded {len(gdf)} features from {country_name} ({filename})")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            missing_files.append((country_name, filename))

    if missing_files:
        print(f"\nWarning: {len(missing_files)} files were missing or failed to load:")
        for country_name, filename in missing_files:
            print(f"  - {country_name}: {filename}")

    print(f"\nSuccessfully loaded {len(gdfs)} country/territory files")
    return gdfs


def combine_and_assign_ids(gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """Combine all GeoDataFrames and assign unique region IDs."""
    if not gdfs:
        raise ValueError("No GeoDataFrames to combine")

    print("Combining all regions into single dataset...")

    # Combine all GeoDataFrames
    combined_gdf = pd.concat(gdfs, ignore_index=True)

    print(f"Combined dataset contains {len(combined_gdf)} total features")

    # Filter out waterbody regions
    initial_count = len(combined_gdf)
    waterbody_mask = (
        (combined_gdf.get("TYPE_1", "").str.contains("Waterbody", case=False, na=False))
        | (
            combined_gdf.get("ENGTYPE_1", "").str.contains(
                "Waterbody", case=False, na=False
            )
        )
        | (
            combined_gdf.get("TYPE_2", "").str.contains(
                "Waterbody", case=False, na=False
            )
        )
        | (
            combined_gdf.get("ENGTYPE_2", "").str.contains(
                "Waterbody", case=False, na=False
            )
        )
    )

    combined_gdf = combined_gdf[~waterbody_mask].copy()
    filtered_count = initial_count - len(combined_gdf)

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} waterbody regions")

    print(f"Final dataset contains {len(combined_gdf)} features after filtering")

    # Assign unique region IDs (1-based indexing)
    combined_gdf["region_id"] = range(1, len(combined_gdf) + 1)

    # Create a human-readable region identifier
    combined_gdf["region_identifier"] = (
        combined_gdf["source_country_code"]
        + "_"
        + combined_gdf["source_admin_level"].astype(str)
        + "_"
        + combined_gdf.index.astype(str)
    )

    # Reorder columns to put ID fields first
    id_cols = [
        "region_id",
        "region_identifier",
        "source_country_code",
        "source_country_name",
        "source_admin_level",
        "source_filename",
    ]
    other_cols = [
        col for col in combined_gdf.columns if col not in id_cols and col != "geometry"
    ]
    column_order = id_cols + other_cols + ["geometry"]

    combined_gdf = combined_gdf[column_order]

    print(f"Assigned unique IDs from 1 to {len(combined_gdf)}")
    return combined_gdf


def save_global_regions(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Save the global regions dataset to a GeoJSON file."""
    print(f"Saving global regions dataset to {output_path}...")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as GeoJSON
    gdf.to_file(output_path, driver="GeoJSON")

    # Create a summary metadata file
    metadata_path = output_path.replace(".geojson", "_metadata.json")
    metadata = {
        "total_regions": len(gdf),
        "countries_included": gdf["source_country_name"].nunique(),
        "admin_levels": sorted(gdf["source_admin_level"].unique().tolist()),
        "region_id_range": {
            "min": int(gdf["region_id"].min()),
            "max": int(gdf["region_id"].max()),
        },
        "columns": gdf.columns.tolist(),
        "crs": str(gdf.crs) if gdf.crs else None,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Global regions dataset saved successfully:")
    print(f"  - Main file: {output_path}")
    print(f"  - Metadata: {metadata_path}")
    print(f"  - Total regions: {len(gdf)}")
    print(f"  - Countries/territories: {gdf['source_country_name'].nunique()}")


def create_global_regions(
    csv_path: str | None = None,
    gadm_dir: str = "country-bounds",
    output_path: str = "global_regions.geojson",
    download_missing: bool = True,
    skip_existing_downloads: bool = True,
) -> bool:
    """
    Main function to create global regions file with unique IDs.

    Args:
        csv_path: Path to countries configuration CSV
        gadm_dir: Directory containing GADM files
        output_path: Path for output GeoJSON file
        download_missing: Whether to download missing GADM files
        skip_existing_downloads: Whether to skip existing files when downloading

    Returns:
        True if successful, False otherwise
    """
    print("=== Creating Global Regions with Unique IDs ===\n")

    # Get CSV path
    if csv_path is None:
        csv_path = get_default_csv_path()
        if csv_path is None:
            return False

    # Read countries configuration
    countries_df = read_countries_config(csv_path)
    if countries_df is None:
        return False

    # Download GADM data if requested
    if download_missing:
        print("Downloading GADM data...")
        success = download_gadm_data(
            csv_path=csv_path,
            download_dir=gadm_dir,
            skip_existing=skip_existing_downloads,
        )
        if not success:
            print("Failed to download GADM data")
            return False
        print()

    # Load GADM files
    gdfs = load_gadm_files(countries_df, gadm_dir)
    if not gdfs:
        print("No GADM files were successfully loaded")
        return False

    # Combine and assign IDs
    try:
        global_gdf = combine_and_assign_ids(gdfs)
    except Exception as e:
        print(f"Error combining regions: {e}")
        return False

    # Save results
    try:
        save_global_regions(global_gdf, output_path)
    except Exception as e:
        print(f"Error saving global regions: {e}")
        return False

    print("\n=== Global regions creation completed successfully! ===")
    return True


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create global regions file with unique IDs from GADM data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c", "--csv", dest="csv_path", help="Path to countries configuration CSV file"
    )

    parser.add_argument(
        "-g",
        "--gadm-dir",
        default="country-bounds",
        help="Directory containing GADM GeoJSON files",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="global_regions.geojson",
        help="Output path for global regions GeoJSON file",
    )

    parser.add_argument(
        "--no-download", action="store_true", help="Skip downloading missing GADM files"
    )

    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Overwrite existing GADM files when downloading",
    )

    return parser.parse_args()


def cli() -> None:
    """Command-line interface entry point."""
    args = parse_arguments()

    success = create_global_regions(
        csv_path=args.csv_path,
        gadm_dir=args.gadm_dir,
        output_path=args.output,
        download_missing=not args.no_download,
        skip_existing_downloads=not args.overwrite_downloads,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    cli()
