import pytest

from BaseITS.pre_processing import *

# Making the module only accept wide_dfs


# def test_aggregation_long_df_type(long_df):
#     """Tests to be written for the upper classes, not base class"""
#     try:
#         long_aggregated_df = aggregation_long_df_type(
#             df=long_df,
#             location_col_name="region",
#             date_col_name="ds",
#             outcome_col_name="outcome",
#             outcome_value_col_name="value",
#         )
#         assert long_aggregated_df.shape[0] != long_df.shape[0]

#     except Exception as e:
#         print(f"Encountered error {e} while aggregating long df type")
#         assert False, "Call to `aggregation_long_df_type` failed"


def test_aggregation_wide_df_type(wide_df, outcomes):
    """Tests to be written for the upper classes, not base class"""
    try:
        wide_df = dates_validation(wide_df, "ds")
        wide_aggregated_df = aggregation_wide_df_type(
            df=wide_df,
            location_col_name="location",
            date_col_name="ds",
            outcome_cols=outcomes,
        )
        assert wide_aggregated_df.shape[0] == wide_df.shape[0]

    except Exception as e:
        print(f"Encountered error {e} while aggregating wide df type")
        assert False, "Call to `aggregation_wide_df_type` failed"


def test_dates_validation(prophet_data, X):
    try:
        date_series = dates_validation(prophet_data, X)
        assert date_series[X].dtype == "datetime64[ns]"

    except Exception as e:
        print(f"Encountered error {e} while validating dates")
        assert False, "Call to `dates_validation` failed"


def test_align_prophet_naming_convection(prophet_data: pd.DataFrame, X: str, y: str):
    try:
        df = align_prophet_naming_convection(prophet_data, X, y)
        column_list = df.columns.tolist()

        assert {"ds", "y"}.issubset(set(column_list))

    except Exception as e:
        print(f"Encountered error {e} while testing prophet naming convection")
        assert False, "Call to `align_prophet_naming_convection` failed"
