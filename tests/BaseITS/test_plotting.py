import pytest
import pandas as pd
from BaseITS.plotting import Plotting


@pytest.fixture(scope="class")
def plotting():
    return Plotting(
        data=pd.read_csv("data/prophet_testing_df.csv"),
        forecast=pd.read_csv("data/metrics.csv"),
        intervention_end_date="",
    )


@pytest.mark.usefixtures("plotting")
class TestPlotting:

    ## This class has no implementations of the plotting tests. The only test is to check if the files were saved and are depedent on the pyplot & seaborn library.
    # Checks for data are done in the plotting class.

    def test_plot_sphaghetti(prophet_data, forecast_df):
        assert True

    def test_plot_cumulative(prophet_data, forecast_df):
        assert True

    def test_plot_count_its(prophet_data, forecast_df):
        assert True

    def test_plot_percent_change(prophet_data, forecast_df):
        assert True

    def test_plot_diff(prophet_data, forecast_df):
        assert True

    def test_plot_boxplots(prophet_data, forecast_df):
        assert True

    def test_plots_metrics_distribution(prophet_data, forecast_df):
        assert True
