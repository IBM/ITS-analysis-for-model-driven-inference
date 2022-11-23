import pytest
import pandas as pd

from BaseITS.poisson_regression import PoissonITS
from BaseITS.custom_transform import CustomTransform
from BaseITS.pre_processing import dates_validation, create_log_offset


@pytest.fixture(scope="class")
def poisson_its():
    return PoissonITS()


@pytest.mark.usefixtures("poisson_its")
class TestPoissonITS:
    def test_fit(
        self, poisson_its, poisson_df, X, y, offset_column_name, transform_columns
    ):

        # preprocess the data
        poisson_df = dates_validation(poisson_df, X)
        offset = create_log_offset(poisson_df, offset_column_name)

        # transform the X variable
        transform = CustomTransform(
            columns=transform_columns,
            seasonally_adjusted=True,
            var_name="month",
        )
        X_train = transform.transform(poisson_df)

        poisson_its.fit(X_train, poisson_df[y], offset)

        assert isinstance(poisson_its, PoissonITS)

    def test_predict(
        self,
        poisson_its,
        factor,
        X,
        poisson_df,
        transform_columns,
        y,
        offset_column_name,
    ):

        poisson_df = dates_validation(poisson_df, X)

        # Create y variable based on the population & Diabetes outcome
        poisson_df["y"] = poisson_df[y] / poisson_df[offset_column_name]

        # transform the X variable
        transform = CustomTransform(
            columns=transform_columns,
            seasonally_adjusted=True,
            var_name="month",
        )

        X_test = transform.transform(poisson_df)

        forecast = poisson_its.predict(X_test, poisson_df, factor)
        assert isinstance(forecast, pd.DataFrame) and {
            "ds",
            "y",
            "ds",
            "y",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "change",
            "percent_change",
        }.issubset(set(forecast.columns.tolist()))

    def test_summary(self, poisson_its, capsys):

        value = poisson_its.summary()
        out, err = capsys.readouterr()
        print(value)

        assert "Generalized Linear Model Regression Results" in out
