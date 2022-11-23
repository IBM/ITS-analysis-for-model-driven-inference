import pytest
from BaseITS.metrics import Metrics
from BaseITS.pre_processing import dates_validation


@pytest.fixture(scope="class")
def metrics():
    return Metrics()


@pytest.mark.usefixtures("metrics")
class TestMetrics:
    def test_get_forecast_metrics(
        self, forecast_df: dict, metrics_columns: list, metrics: Metrics, X: str
    ):

        forecast_df = dates_validation(forecast_df, date_col_name=X)
        results = metrics.get_forecast_metrics(
            forecast=forecast_df,
            location="Gotham",
            outcome="X1",
            prediction_start_date="2020-04-30",
            prediction_end_date="2020-12-30",
        )

        assert set(metrics_columns).issubset(list(results.keys()))
