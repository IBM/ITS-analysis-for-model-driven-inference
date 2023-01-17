# from prophet import *
import pytest
import pandas as pd
import os

# Clean up function to remove files created during tests
def delete_if_exists(file_list):
    """Function to delete a file if it exists

    Args:
        file_list (any): Files list to be removed.
    """
    for f in file_list:
        if os.path.isfile(f):
            os.remove(f)


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Fixture to execute code before and after a test is run"""
    file_list = [file for file in os.listdir("tests/plots") if file.endswith(".png")]
    # Before test - If files exist, delete
    delete_if_exists(file_list)
    yield  # this is where the actual testing happens
    # After test - If files exist, delete
    delete_if_exists(file_list)


@pytest.fixture
def poisson_df() -> pd.DataFrame:
    """Fixture with the poisson dataframe used for testing

    Returns:
        pd.DataFrame: poisson specific dataframe
    """
    return pd.read_csv("data/poisson_testing_df.csv")


@pytest.fixture
def prophet_data() -> pd.DataFrame:
    """Fixture with the prophet dataframe used for testing

    Returns:
        pd.DataFrame: prophet specific dataframe
    """
    return pd.read_csv("data/prophet_testing_df.csv")


# Making the module only accept wide_dfs
# @pytest.fixture
# def long_df():
#     return pd.read_csv("data/long_data_format.csv")


@pytest.fixture
def wide_df():
    """Fixture with the wide type dataframe used for testing

    Returns:
        _type_: wide type dataframe
    """
    return pd.read_csv("data/wide_format_data.csv")


@pytest.fixture
def X() -> str:
    """Fixture for the X, the datetime column"""
    return "ds"


@pytest.fixture
def y() -> str:
    """Fixture for the y, the outcome column"""
    return "X1"


@pytest.fixture
def offset_column_name() -> str:
    """Fixture for the offset, the offset column"""
    return "pop"


@pytest.fixture
def factor() -> int:
    """Fixture for the factor, used in poisson transformation"""
    return 100000


@pytest.fixture
def transform_columns() -> list:
    """Fixture for the harmonic poisson columns"""
    return ["pandemic_april", "time"]


@pytest.fixture
def outcomes() -> str:
    """Fixture for the outcome columns"""
    return ["X1", "X2", "X3", "X4"]


@pytest.fixture
def metrics_columns() -> list:
    """Fixture for the metrics columns"""
    return [
        "location",
        "outcome",
        "mape_before",
        "mape_after",
        "actual_mean_before",
        "predicted_mean_before",
        "actual_mean_after",
        "predicted_mean_after",
        "actual_median_before",
        "predicted_median_before",
        "actual_median_after",
        "predicted_median_after",
        "mean_change_before",
        "wilcoxon_change_before",
        "mean_change_after",
        "wilcoxon_change_after",
        "change_conf_int_before",
        "change_conf_int_after",
        "mean_percent_change_before",
        "wilcoxon_percent_change_before",
        "mean_percent_change_after",
        "wilcoxon_percent_change_after",
        "percent_change_conf_int_before",
        "percent_change_conf_int_after",
    ]


@pytest.fixture
def params() -> dict():
    """Fixture for the parameters values to be tuned. (dict)"""
    return {
        "changepoint_prior_scale": [0.001],
        "seasonality_prior_scale": [0.1],
        "seasonality_mode": ["additive", "multiplicative"],
    }


@pytest.fixture
def forecast_df() -> pd.DataFrame:
    """Fixture with the forecast dataframe used for testing

    Returns:
        pd.DataFrame: forecast dataframe with the outputs of the models
    """
    return pd.read_csv("data/metrics.csv")
