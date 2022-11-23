import pytest
from BaseITS.model_tuning import ModelTuning
import pandas as pd
from BaseITS.pre_processing import *


@pytest.fixture(scope="class")
def tuning():
    return ModelTuning()


@pytest.mark.usefixtures("tuning")
class TestModelTuning:
    # takes too long to run, commented out.
    # TODO Maybe use pool?
    pass
    # def test_tune_hyperparameters(self, tuning:ModelTuning(), prophet_data:pd.DataFrame, params:dict, X:str, y:str):

    #     # prophet_data =
    #     # prediction_start_date = datetime.strptime("2020-04-30", "%Y-%m-%d")
    #     prophet_data = dates_validation(prophet_data, X)
    #     prophet_data = align_prophet_naming_convection(prophet_data, X, y)

    #     tuning_results, best_params = tuning.tune_hyperparameters(
    #         df=prophet_data, param_grid=params
    #     )
    #     assert (
    #         isinstance(tuning_results, pd.DataFrame)
    #         and set(["rmse"]).isubset(tuning_results.columns.tolist())
    #         and isinstance(best_params, dict)
    #     )
