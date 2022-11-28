from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, sem, t

from BaseITS.utils import *


class Metrics:
    """Class to generate metrics from the outputs of the models.

    def get_forecast_metrics()
        Returns a dictionary with the following metrics:
            mape_before: Mean absolute percentage error before the intervention
            mape_after: Mean absolute percentage error after the intervention

            actual_mean_before: Actual mean before the intervention
            predicted_mean_before: Predicted mean before the intervention
            actual_mean_after: Actual mean after the intervention
            predicted_mean_after: Predicted mean after the intervention

            actual_median_before: Actual median before the intervention
            predicted_median_before: Predicted median before the intervention
            actual_median_after: Actual median after the intervention
            predicted_median_after: Predicted median after the intervention

            mean_change_before: Mean change before the intervention
            wilcoxon_change_before: Wilcoxon change before the intervention
            mean_change_after: Mean change after the intervention
            wilcoxon_change_after: Wilcoxon change afte the intervention

            change_conf_int_before: Confidence interval change before the intervention
            change_conf_int_after: Confidence interval change before the intervention

            mean_percent_change_before: Mean percentage change before the intervention
            wilcoxon_percent_change_before: Wilcoxon percentage change before the intervention
            mean_percent_change_after: Mean percentage change after the intervention
            wilcoxon_percent_change_after: Wilcoxon percentage change after the intervention

            percent_change_conf_int_before: Confidence interval percentage change before the intervention
            percent_change_conf_int_after: Confidence interval percentage change after the intervention

    """

    # TODO: Add a validate inputs function to make sure the inputs passed are correct.

    def __init__(self) -> None:
        pass

    def get_forecast_metrics(
        self,
        location: str,
        outcome: str,
        forecast: dict(),
        prediction_start_date: str,
        prediction_end_date: str,
    ):
        """Function to get the metrics from the forecast.


        Args:
            location (str): Geographical unit
            outcome (str):  Outcome measure
            forecast (dict): dictionary with the prophet forecast output for the geographical unit and outcome
            prediction_start_date (datetime): Prediction start date
            prediction_end_date (datetime): Prediction end date

        Returns:
            dict: Dictionary with generated metrics
        """

        prediction_start_date = datetime.strptime(prediction_start_date, "%Y-%m-%d")
        prediction_end_date = datetime.strptime(prediction_end_date, "%Y-%m-%d")

        df_before = forecast[forecast["ds"] < prediction_start_date]

        df_after = forecast[
            (forecast["ds"] >= prediction_start_date)
            & (forecast["ds"] <= prediction_end_date)
        ]

        metrics = dict()
        metrics["location"] = location
        metrics["outcome"] = outcome

        metrics["mape_before"] = round(
            np.mean(((df_before["y"] - df_before["yhat"]) / df_before["y"]).abs()), 2
        )
        metrics["mape_after"] = round(
            np.mean(((df_after["y"] - df_after["yhat"]) / df_after["y"]).abs()), 2
        )

        metrics["actual_mean_before"] = int(round(df_before["y"].mean(), 0))
        metrics["predicted_mean_before"] = int(round(df_before["yhat"].mean(), 0))
        metrics["actual_mean_after"] = int(round(df_after["y"].mean(), 0))
        metrics["predicted_mean_after"] = int(round(df_after["yhat"].mean(), 0))

        metrics["actual_median_before"] = int(round(df_before["y"].median(), 0))
        metrics["predicted_median_before"] = int(round(df_before["yhat"].median(), 0))
        metrics["actual_median_after"] = int(round(df_after["y"].median(), 0))
        metrics["predicted_median_after"] = int(round(df_after["yhat"].median(), 0))

        metrics["mean_change_before"] = np.mean(df_before["change"])
        metrics["wilcoxon_change_before"] = wilcoxon(df_before["change"])
        metrics["mean_change_after"] = np.mean(df_after["change"])
        metrics["wilcoxon_change_after"] = wilcoxon(df_after["change"])

        metrics["change_conf_int_before"] = t.interval(
            alpha=0.95,
            df=len(df_before["change"]) - 1,
            loc=np.mean(df_before["change"]),
            scale=sem(df_before["change"]),
        )
        metrics["change_conf_int_after"] = t.interval(
            alpha=0.95,
            df=len(df_after["change"]) - 1,
            loc=np.mean(df_after["change"]),
            scale=sem(df_after["change"]),
        )

        metrics["mean_percent_change_before"] = np.mean(df_before["percent_change"])
        metrics["wilcoxon_percent_change_before"] = wilcoxon(
            df_before["percent_change"]
        )
        metrics["mean_percent_change_after"] = np.mean(df_after["percent_change"])
        metrics["wilcoxon_percent_change_after"] = wilcoxon(df_after["percent_change"])

        metrics["percent_change_conf_int_before"] = t.interval(
            alpha=0.95,
            df=len(df_before["percent_change"]) - 1,
            loc=np.mean(df_before["percent_change"]),
            scale=sem(df_before["percent_change"]),
        )
        metrics["percent_change_conf_int_after"] = t.interval(
            alpha=0.95,
            df=len(df_after["percent_change"]) - 1,
            loc=np.mean(df_after["percent_change"]),
            scale=sem(df_after["percent_change"]),
        )

        return metrics
