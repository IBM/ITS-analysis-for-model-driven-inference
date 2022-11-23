import pytest
import sys

from BaseITS.utils import *

# from Interrupted_Time_Series_Prophet import *


def test_check_dataset_format(wide_df, outcomes):

    try:
        dataset_format = check_dataset_format(wide_df, outcomes)
        print(dataset_format, "dataset_format")
        assert dataset_format in ["wide", "long"]

    except Exception as e:
        print(f"Encountered error {e} while checking the dataset format")
        assert False, "Call to `check_dataset_format` failed"


# def test_extract_inputs(forecast_dict: dict):
#     try:
#         extracted_values = extract_inputs(forecast_dict)
#         assert isinstance(extracted_values, tuple)
#     except Exception as e:
#         print(f"Encountered error {e} while extracting forecast data")
#         assert False, "Call to `extract_inputs` failed"
