import pytest

import pandas as pd
from BaseITS.wrapper_class import BaseITS
from BaseITS.prophet_model import ProphetITS
from BaseITS.pre_processing import dates_validation


@pytest.fixture(scope="class")
def wrapper():
    return BaseITS(
        model=["prophet"],
        location=["Gotham"],
        outcome=["X1"],
        interruption_date=["2020-04-30"],
    )


@pytest.mark.usefixtures("wrapper")
class TestBaseITS:
    def test_fit(self, wrapper: BaseITS, prophet_data: pd.DataFrame):

        prophet_data = dates_validation(prophet_data, "ds")
        one_region_df = prophet_data[prophet_data.location == "Gotham"]
        one_region_df = one_region_df[["ds", "location", "X1"]]

        fitted_model = wrapper.fit(
            df=one_region_df, X=one_region_df["ds"], y=one_region_df["X1"]
        )

        assert isinstance(fitted_model, ProphetITS)

    def test_predict(self, wrapper: BaseITS, prophet_data: pd.DataFrame):

        prophet_data = dates_validation(prophet_data, "ds")
        one_region_df = prophet_data[prophet_data.location == "Gotham"]
        one_region_df = one_region_df[["ds", "location", "X1"]]
        results = wrapper.predict(
            df=one_region_df,
            X=one_region_df["ds"],
            y=one_region_df["X1"],
        )
        print(results.columns)
        assert isinstance(results, pd.DataFrame) and {
            "change",
            "percent_change",
            "y",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        }.issubset(set(results.columns.tolist()))

    def test_fit_predict(self, wrapper: BaseITS, prophet_data: pd.DataFrame):
        prophet_data = dates_validation(prophet_data, "ds")
        one_region_df = prophet_data[prophet_data.location == "Gotham"]
        one_region_df = one_region_df[["ds", "location", "X1"]]

        ## TODO Figure out why the predict function is not called here.
        results = wrapper.fit_predict(
            df=one_region_df,
            X=one_region_df["ds"],
            y=one_region_df["X1"],
        )
        assert isinstance(results, pd.DataFrame) and {
            "change",
            "percent_change",
            "y",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        }.issubset(set(results.columns.tolist()))
