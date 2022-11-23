import pytest
from datetime import datetime

from prophet import Prophet
from BaseITS.prophet_model import ProphetITS
from BaseITS.pre_processing import *


@pytest.fixture(scope="class")
def prophet():
    return ProphetITS()


@pytest.mark.usefixtures("prophet")
class TestProphetITS:
    # def __init__(self) -> None:
    #     assert False

    def test_fit(self, X: str, y: str, prophet_data: pd.DataFrame, prophet: ProphetITS):

        # Creating a training dataframe from the whole prophet dataset.
        # Pre-processing the dataset before fitting.
        prediction_start_date = datetime.strptime("2020-04-30", "%Y-%m-%d")
        prophet_data = dates_validation(prophet_data, X)
        prophet_data = prophet_data[prophet_data["ds"] <= prediction_start_date]

        prophet_data = align_prophet_naming_convection(prophet_data, X, y)

        fitted_model = prophet.fit(prophet_data)

        assert isinstance(fitted_model, Prophet)

    def test_predict(
        self, prophet: ProphetITS, prophet_data: pd.DataFrame, X: str, y: str
    ) -> pd.DataFrame:

        # Pre-processing the dataset before prediction.
        prophet_data = dates_validation(prophet_data, X)
        prophet_data = align_prophet_naming_convection(prophet_data, X, y)

        results = prophet.predict(prophet_data)

        assert {
            "change",
            "percent_change",
            "y",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        }.issubset(set(results.columns.tolist())) and isinstance(results, pd.DataFrame)

    def test_fit_predict(
        self, prophet: ProphetITS, prophet_data: pd.DataFrame, X: str, y: str
    ):
        prediction_start_date = datetime.strptime("2020-04-30", "%Y-%m-%d")
        prophet_data = dates_validation(prophet_data, X)
        prophet_data = prophet_data[prophet_data["ds"] <= prediction_start_date]

        prophet_data = align_prophet_naming_convection(prophet_data, X, y)

        predictions = prophet.fit_predict(df=prophet_data)
        print(type(predictions))
        print(predictions.columns.tolist())

        assert set(
            [
                "change",
                "percent_change",
                "y",
                "yhat",
                "yhat_lower",
                "yhat_upper",
            ]
        ).issubset(set(predictions.columns.tolist())) and isinstance(
            predictions, pd.DataFrame
        )

    def test_summary(self, prophet: ProphetITS):
        value = prophet.summary()

        assert isinstance(value, str)
