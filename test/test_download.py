from botocore import UNSIGNED
import pytest
from unittest.mock import patch, MagicMock, call, mock_open

# Import the module to be tested
from cmip6atlas.download import (
    create_s3_client,
    get_available_models,
    get_available_files,
    download_file,
    download_files_parallel,
    format_size,
    validate_inputs,
    download_granules,
    GranuleSubset,
    DownloadError,
    GDDP_CMIP6_SCHEMA,
)


class TestCreateS3Client:
    def test_create_s3_client(self):
        """Test that create_s3_client returns an S3 client with unsigned config."""
        with patch("boto3.client") as mock_boto3_client:
            create_s3_client()
            mock_boto3_client.assert_called_once()
            # Check that it was called with unsigned config
            _, kwargs = mock_boto3_client.call_args
            assert "config" in kwargs
            assert kwargs["config"].signature_version == UNSIGNED


class TestGetAvailableModels:
    def test_get_available_models_success(self):
        """Test successful retrieval of available models."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "CommonPrefixes": [
                {"Prefix": "NEX-GDDP-CMIP6/ACCESS-CM2/"},
                {"Prefix": "NEX-GDDP-CMIP6/CESM2/"},
                {"Prefix": "NEX-GDDP-CMIP6/MIROC6/"},
            ]
        }

        models = get_available_models(mock_client, "test-bucket", "NEX-GDDP-CMIP6")
        
        assert models == ["ACCESS-CM2", "CESM2", "MIROC6"]
        mock_client.list_objects_v2.assert_called_once_with(
            Bucket="test-bucket", Prefix="NEX-GDDP-CMIP6", Delimiter="/"
        )

    def test_get_available_models_no_models(self):
        """Test DownloadError is raised when no models are found."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {"CommonPrefixes": []}

        with pytest.raises(DownloadError) as excinfo:
            get_available_models(mock_client, "test-bucket", "NEX-GDDP-CMIP6")
        
        assert "could not retrieve available models" in str(excinfo.value)


class TestGetAvailableFiles:
    def test_get_available_files_one_scenario(self):
        """Test retrieval of files for a single scenario."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/ssp585/r1i1p1f1/tas/tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2025.nc",
                    "Size": 1024000,
                },
                {
                    "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/ssp585/r1i1p1f1/tas/tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2026.nc",
                    "Size": 1024000,
                },
                {
                    "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/ssp585/r1i1p1f1/tas/tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2027.nc",
                    "Size": 1024000,
                },
            ]
        }

        schema = GDDP_CMIP6_SCHEMA.copy()
        files = get_available_files(
            mock_client, schema, "ACCESS-CM2", "ssp585", "tas", 2025, 2026
        )

        # Should only get files for 2025 and 2026
        assert len(files) == 2
        assert files[0][1] == "tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2025.nc"
        assert files[1][1] == "tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2026.nc"
    
    def test_get_available_files_historical(self):
        """Test retrieval of files for a single scenario."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tas/tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_1991.nc",
                    "Size": 1024000,
                },
                {
                    "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tas/tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_1992.nc",
                    "Size": 1024000,
                },
            ]
        }

        schema = GDDP_CMIP6_SCHEMA.copy()
        files = get_available_files(
            mock_client, schema, "ACCESS-CM2", "ssp585", "tas", 1991, 1992
        )

        # Should get files from historical despite using ssp585 scenario
        assert len(files) == 2
        assert files[0][1] == "tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_1991.nc"
        assert files[1][1] == "tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_1992.nc"

    def test_get_available_files_cross_historical(self):
        """Test retrieval of files spanning historical and projected periods."""
        mock_client = MagicMock()
        
        # First call for historical data
        mock_client.list_objects_v2.side_effect = [
            {
                "Contents": [
                    {
                        "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tas/tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_2013.nc",
                        "Size": 1024000,
                    },
                    {
                        "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/historical/r1i1p1f1/tas/tas_day_ACCESS-CM2_historical_r1i1p1f1_gn_2014.nc",
                        "Size": 1024000,
                    },
                ]
            },
            # Second call for projected data
            {
                "Contents": [
                    {
                        "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/ssp585/r1i1p1f1/tas/tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2015.nc",
                        "Size": 1024000,
                    },
                    {
                        "Key": "NEX-GDDP-CMIP6/ACCESS-CM2/ssp585/r1i1p1f1/tas/tas_day_ACCESS-CM2_ssp585_r1i1p1f1_gn_2016.nc",
                        "Size": 1024000,
                    },
                ]
            },
        ]

        schema = GDDP_CMIP6_SCHEMA.copy()
        schema["historical_end_year"] = 2014  # Ensure this is set correctly
        
        files = get_available_files(
            mock_client, schema, "ACCESS-CM2", "ssp585", "tas", 2013, 2016
        )

        # Should get 4 files spanning historical and projected periods
        assert len(files) == 4
        assert "historical" in files[0][0]
        assert "historical" in files[1][0]
        assert "ssp585" in files[2][0]
        assert "ssp585" in files[3][0]

    def test_get_available_files_version_handling(self):
        """Test handling of file versions."""
        mock_client = MagicMock()
        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {
                    "Key": "NEX-GDDP-CMIP6/TaiESM1/ssp585/r1i1p1f1/tas/tas_day_TaiESM1_ssp585_r1i1p1f1_gn_2015.nc",
                    "Size": 1024000,
                },
                {
                    "Key": "NEX-GDDP-CMIP6/TaiESM1/ssp585/r1i1p1f1/tas/tas_day_TaiESM1_ssp585_r1i1p1f1_gn_2015_v1.1.nc",
                    "Size": 1024000,
                },
                {
                    "Key": "NEX-GDDP-CMIP6/TaiESM1/ssp585/r1i1p1f1/tas/tas_day_TaiESM1_ssp585_r1i1p1f1_gn_2015_v1.2.nc",
                    "Size": 1024000,
                },
            ]
        }

        schema = GDDP_CMIP6_SCHEMA.copy()
        files = get_available_files(
            mock_client, schema, "TaiESM1", "ssp585", "tas", 2015, 2015
        )

        # Should only get the latest version
        assert len(files) == 1
        assert files[0][1] == "tas_day_TaiESM1_ssp585_r1i1p1f1_gn_2015_v1.2.nc"


class TestDownloadFile:
    def test_download_file_new(self):
        """Test downloading a new file."""
        mock_client = MagicMock()
        
        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs") as mock_makedirs, \
             patch("os.path.dirname", return_value="/test/dir"):
            
            result = download_file(
                mock_client, "test-bucket", "test-key", "/test/dir/test-file.nc"
            )
            
            mock_makedirs.assert_called_once_with("/test/dir")
            mock_client.download_file.assert_called_once_with(
                "test-bucket", "test-key", "/test/dir/test-file.nc"
            )
            assert result is True

    def test_download_file_exists_same_size(self):
        """Test skipping an existing file with the same size."""
        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 1024000}
        
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=1024000):
            
            result = download_file(
                mock_client, "test-bucket", "test-key", "/test/dir/test-file.nc"
            )
            
            mock_client.head_object.assert_called_once_with(
                Bucket="test-bucket", Key="test-key"
            )
            mock_client.download_file.assert_not_called()
            assert result is False

    def test_download_file_exists_different_size(self):
        """Test re-downloading an existing file with a different size."""
        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ContentLength": 1024000}
        
        with patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=512000), \
             patch("os.makedirs") as mock_makedirs, \
             patch("os.path.dirname", return_value="/test/dir"):
            
            result = download_file(
                mock_client, "test-bucket", "test-key", "/test/dir/test-file.nc"
            )
            
            mock_client.head_object.assert_called_once_with(
                Bucket="test-bucket", Key="test-key"
            )
            mock_client.download_file.assert_called_once_with(
                "test-bucket", "test-key", "/test/dir/test-file.nc"
            )
            assert result is True


class TestDownloadFilesParallel:
    def test_download_files_parallel(self):
        """Test parallel downloading of multiple files."""
        mock_client = MagicMock()
        files = [
            ("key1", "file1.nc", 1024),
            ("key2", "file2.nc", 2048),
            ("key3", "file3.nc", 3072),
        ]
        
        # Mock the concurrent.futures.ThreadPoolExecutor
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor, \
             patch("os.path.join", side_effect=lambda d, f: f"{d}/{f}"):
            
            # Mock executor's submit method and future results
            mock_executor_instance = mock_executor.return_value.__enter__.return_value
            mock_future1, mock_future2, mock_future3 = MagicMock(), MagicMock(), MagicMock()
            mock_future1.result.return_value = True
            mock_future2.result.return_value = False
            mock_future3.result.return_value = True
            mock_executor_instance.submit.side_effect = [mock_future1, mock_future2, mock_future3]
            
            # Mock as_completed to return futures in order
            with patch("concurrent.futures.as_completed", 
                      return_value=[mock_future1, mock_future2, mock_future3]):
                
                result = download_files_parallel(
                    mock_client, "test-bucket", files, "/output", 3
                )
            
            # Check that submit was called correctly for each file
            assert mock_executor_instance.submit.call_count == 3
            expected_calls = [
                call(download_file, mock_client, "test-bucket", "key1", "/output/file1.nc"),
                call(download_file, mock_client, "test-bucket", "key2", "/output/file2.nc"),
                call(download_file, mock_client, "test-bucket", "key3", "/output/file3.nc"),
            ]
            mock_executor_instance.submit.assert_has_calls(expected_calls)
            
            # 2 files should have been downloaded (first and third returned True)
            assert result == 2


class TestFormatSize:
    def test_format_size_bytes(self):
        """Test formatting small file sizes in bytes."""
        assert format_size(500) == "500.00 B"

    def test_format_size_kilobytes(self):
        """Test formatting file sizes in kilobytes."""
        assert format_size(1500) == "1.46 KB"

    def test_format_size_megabytes(self):
        """Test formatting file sizes in megabytes."""
        assert format_size(1500000) == "1.43 MB"

    def test_format_size_gigabytes(self):
        """Test formatting file sizes in gigabytes."""
        assert format_size(1500000000) == "1.40 GB"

    def test_format_size_terabytes(self):
        """Test formatting file sizes in terabytes."""
        assert format_size(1500000000000) == "1.36 TB"


class TestValidateInputs:
    def test_valid_inputs(self):
        """Test validation with valid inputs."""
        schema = GDDP_CMIP6_SCHEMA.copy()
        inputs = GranuleSubset(
            start_year=2020,
            end_year=2030,
            variable="tas",
            scenario="ssp585",
            models=["ACCESS-CM2", "CESM2"]
        )
        
        # Should not raise any exceptions
        validate_inputs(schema, inputs)

    def test_invalid_start_year(self):
        """Test validation with invalid start year."""
        schema = GDDP_CMIP6_SCHEMA.copy()
        inputs = GranuleSubset(
            start_year=1900,  # Before min_year
            end_year=2030,
            variable="tas",
            scenario="ssp585",
            models=["ACCESS-CM2"]
        )
        
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(schema, inputs)
        assert "Start year" in str(excinfo.value)

    def test_invalid_end_year(self):
        """Test validation with invalid end year."""
        schema = GDDP_CMIP6_SCHEMA.copy()
        inputs = GranuleSubset(
            start_year=2020,
            end_year=2200,  # After max_year
            variable="tas",
            scenario="ssp585",
            models=["ACCESS-CM2"]
        )
        
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(schema, inputs)
        assert "End year" in str(excinfo.value)

    def test_start_after_end(self):
        """Test validation with start year after end year."""
        schema = GDDP_CMIP6_SCHEMA.copy()
        inputs = GranuleSubset(
            start_year=2030,
            end_year=2020,  # Before start_year
            variable="tas",
            scenario="ssp585",
            models=["ACCESS-CM2"]
        )
        
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(schema, inputs)
        assert "Start year cannot be after end year" in str(excinfo.value)

    def test_invalid_variable(self):
        """Test validation with invalid variable."""
        schema = GDDP_CMIP6_SCHEMA.copy()
        inputs = GranuleSubset(
            start_year=2020,
            end_year=2030,
            variable="invalid_var",  # Not in schema
            scenario="ssp585",
            models=["ACCESS-CM2"]
        )
        
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(schema, inputs)
        assert "invalid_var is not available" in str(excinfo.value)

    def test_invalid_scenario(self):
        """Test validation with invalid scenario."""
        schema = GDDP_CMIP6_SCHEMA.copy()
        inputs = GranuleSubset(
            start_year=2020,
            end_year=2030,
            variable="tas",
            scenario="invalid_scenario",  # Not in schema
            models=["ACCESS-CM2"]
        )
        
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(schema, inputs)
        assert "invalid_scenario is not available" in str(excinfo.value)

    def test_historical_out_of_range(self):
        """Test validation with historical scenario out of range."""
        schema = GDDP_CMIP6_SCHEMA.copy()
        schema["historical_end_year"] = 2014
        inputs = GranuleSubset(
            start_year=2010,
            end_year=2020,  # After historical_end_year
            variable="tas",
            scenario="historical",
            models=["ACCESS-CM2"]
        )
        
        with pytest.raises(ValueError) as excinfo:
            validate_inputs(schema, inputs)
        assert "extends outside historical record" in str(excinfo.value)


class TestDownloadGranules:
    @patch("cmip6atlas.download.create_s3_client")
    @patch("cmip6atlas.download.get_available_models")
    @patch("cmip6atlas.download.get_available_files")
    @patch("cmip6atlas.download.download_files_parallel")
    @patch("os.makedirs")
    def test_download_granules_basic(
        self, mock_makedirs, mock_download_parallel, mock_get_files, 
        mock_get_models, mock_create_client
    ):
        """Test basic download granules functionality."""
        # Setup mocks
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_get_models.return_value = ["ACCESS-CM2", "CESM2"]
        mock_get_files.return_value = [
            ("key1", "file1.nc", 1024),
            ("key2", "file2.nc", 2048)
        ]
        mock_download_parallel.return_value = 2
        
        # Call the function
        download_granules(
            variable="tas",
            scenario="ssp585",
            start_year=2020,
            end_year=2021,
            output_dir="/output",
            skip_prompt=True
        )
        
        # Verify the calls
        mock_create_client.assert_called_once()
        mock_get_models.assert_called_once()
        assert mock_get_files.call_count == 2  # Once for each model
        mock_download_parallel.assert_called_once()
        mock_makedirs.assert_called_once_with("/output", exist_ok=True)

    @patch("cmip6atlas.download.create_s3_client")
    @patch("cmip6atlas.download.get_available_models")
    @patch("cmip6atlas.download.get_available_files")
    @patch("cmip6atlas.download.download_files_parallel")
    @patch("os.makedirs")
    @patch("builtins.input", return_value="y")
    def test_download_granules_with_prompt(
        self, mock_input, mock_makedirs, mock_download_parallel, 
        mock_get_files, mock_get_models, mock_create_client
    ):
        """Test download granules with user prompt."""
        # Setup mocks
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_get_models.return_value = ["ACCESS-CM2"]
        mock_get_files.return_value = [("key1", "file1.nc", 1024)]
        mock_download_parallel.return_value = 1
        
        # Call the function
        download_granules(
            variable="tas",
            scenario="ssp585",
            start_year=2020,
            end_year=2020,
            output_dir="/output",
            skip_prompt=False  # Ask for confirmation
        )
        
        # Verify the input prompt was called
        mock_input.assert_called_once()
        mock_download_parallel.assert_called_once()

    @patch("cmip6atlas.download.create_s3_client")
    @patch("cmip6atlas.download.get_available_models")
    @patch("cmip6atlas.download.get_available_files")
    @patch("cmip6atlas.download.download_files_parallel")
    @patch("os.makedirs")
    @patch("builtins.input", return_value="n")
    def test_download_granules_cancel_prompt(
        self, mock_input, mock_makedirs, mock_download_parallel, 
        mock_get_files, mock_get_models, mock_create_client
    ):
        """Test canceling download from prompt."""
        # Setup mocks
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_get_models.return_value = ["ACCESS-CM2"]
        mock_get_files.return_value = [("key1", "file1.nc", 1024)]
        
        # Call the function
        download_granules(
            variable="tas",
            scenario="ssp585",
            start_year=2020,
            end_year=2020,
            output_dir="/output",
            skip_prompt=False  # Ask for confirmation
        )
        
        # Verify the download was not called
        mock_input.assert_called_once()
        mock_download_parallel.assert_not_called()

    @patch("cmip6atlas.download.create_s3_client")
    @patch("cmip6atlas.download.get_available_models")
    @patch("cmip6atlas.download.get_available_files")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_download_granules_all_exist(
        self, mock_getsize, mock_exists, mock_get_files, 
        mock_get_models, mock_create_client
    ):
        """Test when all files already exist."""
        # Setup mocks
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_get_models.return_value = ["ACCESS-CM2"]
        mock_get_files.return_value = [("key1", "file1.nc", 1024)]
        mock_exists.return_value = True
        mock_getsize.return_value = 1024  # Same size as in mock_get_files
        
        # Call the function
        with patch("builtins.print") as mock_print:
            download_granules(
                variable="tas",
                scenario="ssp585",
                start_year=2020,
                end_year=2020,
                output_dir="/output",
                skip_prompt=True
            )
            
            # Check that the appropriate message was printed
            mock_print.assert_any_call("\nAll files already exist locally. No downloads needed.")

    @patch("cmip6atlas.download.create_s3_client")
    @patch("cmip6atlas.download.get_available_models")
    @patch("cmip6atlas.download.get_available_files")
    def test_download_granules_no_files_found(
        self, mock_get_files, mock_get_models, mock_create_client
    ):
        """Test when no files match the criteria."""
        # Setup mocks
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_get_models.return_value = ["ACCESS-CM2"]
        mock_get_files.return_value = []  # No files found
        
        # Call the function
        with patch("builtins.print") as mock_print:
            download_granules(
                variable="tas",
                scenario="ssp585",
                start_year=2020,
                output_dir="/output",
                skip_prompt=True
            )
            
            # Check that the appropriate message was printed
            mock_print.assert_any_call("No files found matching your criteria. Please check your parameters.")

    @patch("cmip6atlas.download.create_s3_client")
    @patch("cmip6atlas.download.get_available_models")
    def test_download_granules_specific_models(
        self, mock_get_models, mock_create_client
    ):
        """Test download with specific models."""
        # Setup mocks
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_get_models.return_value = ["ACCESS-CM2", "CESM2", "MIROC6"]
        
        # Setup further mocks to prevent execution beyond model filtering
        with patch("cmip6atlas.download.get_available_files", return_value=[]), \
             patch("builtins.print"):
            
            # Verify only valid models are used (intersection with available_models)
            with patch("cmip6atlas.download.validate_inputs") as mock_validate:
                # Call the function with specific models
                download_granules(
                    variable="tas",
                    scenario="ssp585",
                    start_year=2020,
                    output_dir="/output",
                    models=["ACCESS-CM2", "UNKNOWN-MODEL"],  # One valid, one invalid
                    skip_prompt=True
                )
                assert mock_validate.call_args[0][1].models == ["ACCESS-CM2"]

    @patch("cmip6atlas.download.create_s3_client")
    @patch("cmip6atlas.download.get_available_models")
    def test_download_granules_exclude_models(
        self, mock_get_models, mock_create_client
    ):
        """Test download with excluded models."""
        # Setup mocks
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_get_models.return_value = ["ACCESS-CM2", "CESM2", "MIROC6"]
        
        # Setup further mocks to prevent execution beyond model filtering
        with patch("cmip6atlas.download.get_available_files", return_value=[]), \
             patch("builtins.print"):
            
            # Verify the correct models are used (set difference)
            # Use a separate patch to check what gets passed to validate_inputs
            with patch("cmip6atlas.download.validate_inputs") as mock_validate:
                # Call the function with an excluded model
                download_granules(
                    variable="tas",
                    scenario="ssp585",
                    start_year=2020,
                    output_dir="/output",
                    exclude_models=["CESM2"],
                    skip_prompt=True
                )
                # Check that CESM2 is not in the models list
                models_used = mock_validate.call_args[0][1].models
                assert "ACCESS-CM2" in models_used
                assert "MIROC6" in models_used
                assert "CESM2" not in models_used

    def test_download_granules_both_include_exclude(self):
        """Test error when both models and exclude_models are provided."""
        # shoehorn in a schema test for coverage
        schema = GDDP_CMIP6_SCHEMA.copy()
        schema["prefix"] = "NEX-GDDP-CMIP6/"
        with pytest.raises(ValueError) as excinfo:
            download_granules(
                variable="tas",
                scenario="ssp585",
                start_year=2020,
                models=["ACCESS-CM2"],
                exclude_models=["CESM2"],
                schema=schema
            )
        assert "cannot exclude and include models in same query" in str(excinfo.value)


class TestCLI:
    @patch("cmip6atlas.download.download_granules")
    def test_cli_basic(self, mock_download):
        """Test basic CLI functionality."""
        with patch("sys.argv", [
            "script.py",
            "--variable", "tas",
            "--start-year", "2020",
            "--scenario", "ssp585"
        ]):
            from cmip6atlas.download import cli
            cli()
            
            # Verify download_granules was called with correct args
            mock_download.assert_called_once_with(
                "tas", "ssp585", 2020, None, "./nex-gddp-data", None, None, 5, False
            )

    @patch("cmip6atlas.download.download_granules")
    def test_cli_full_options(self, mock_download):
        """Test CLI with all options specified."""
        with patch("sys.argv", [
            "script.py",
            "--variable", "tas",
            "--start-year", "2020",
            "--end-year", "2030",
            "--scenario", "ssp585",
            "--models", "ACCESS-CM2", "CESM2",
            "--output-dir", "/custom/output",
            "--max-workers", "10",
            "--yes"
        ]):
            from cmip6atlas.download import cli
            cli()
            
            # Verify download_granules was called with correct args
            mock_download.assert_called_once_with(
                "tas", "ssp585", 2020, 2030, "/custom/output", 
                ["ACCESS-CM2", "CESM2"], None, 10, True
            )

    @patch("cmip6atlas.download.download_granules")
    def test_cli_exclude_models(self, mock_download):
        """Test CLI with exclude-models option."""
        with patch("sys.argv", [
            "script.py",
            "--variable", "tas",
            "--start-year", "2020",
            "--scenario", "ssp585",
            "--exclude-models", "CESM2", "MIROC6",
            "--yes"
        ]):
            from cmip6atlas.download import cli
            cli()
            
            # Verify download_granules was called with correct args
            mock_download.assert_called_once_with(
                "tas", "ssp585", 2020, None, "./nex-gddp-data", 
                None, ["CESM2", "MIROC6"], 5, True
            )