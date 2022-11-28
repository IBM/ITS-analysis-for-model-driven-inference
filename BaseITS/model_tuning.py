import pandas as pd
import numpy as np
from datetime import datetime
import itertools

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

from BaseITS.pre_processing import str_date_validate


class ModelTuning:
    """Class for tuning Hyperparameters for prophet model. No implementation for Poisson Regression because it's a basic linear regression model.

    Args:
        cutoff_start (str, optional): start date for tuning data . Defaults to "2019-02-28".
        cutoff_end (str, optional): end date for tuning data. Defaults to "2019-10-31".
        param_grid (dict, optional): Dictionary with the parameters to be tuned. Defaults to { "changepoint_prior_scale": [0.001, 0.05], "seasonality_prior_scale": [0.1, 10.0], "seasonality_mode": ["additive", "multiplicative"], }.
    """

    def __init__(
        self,
        # model: Prophet = Prophet(),
        cutoff_start: str = "2019-02-28",
        cutoff_end: str = "2019-10-31",
        param_grid: dict = {
            "changepoint_prior_scale": [0.001, 0.05],
            "seasonality_prior_scale": [0.1, 10.0],
            "seasonality_mode": ["additive", "multiplicative"],
        },
    ):
        """Tuning parameters

        Args:
            cutoff_start (str, optional): start date for tuning data . Defaults to "2019-02-28".
            cutoff_end (str, optional): end date for tuning data. Defaults to "2019-10-31".
            param_grid (dict, optional): Dictionary with the parameters to be tuned. Defaults to { "changepoint_prior_scale": [0.001, 0.05], "seasonality_prior_scale": [0.1, 10.0], "seasonality_mode": ["additive", "multiplicative"], }.
        """

        self.param_grid_ = param_grid
        self.model_ = Prophet()
        self.cutoff_start_ = cutoff_start
        self.cuttoff_end_ = cutoff_end
        self.__validate_inputs()

    def __validate_inputs(self):
        """Function to validate the inputs provided to this class."""
        # confirm no null values are passed. or wrong formats of data

        # str_date_validate(self.cutoff_start_)
        # str_date_validate(self.cuttoff_end_)

        if not (isinstance(self.cutoff_start_, str)) or not (
            isinstance(self.cuttoff_end_, str)
        ):
            raise TypeError(
                '""cutoff_start" and "cutoff_end" date parameters provided must be of type "str"'
            )

        elif not bool(self.cuttoff_end_ and not self.cuttoff_end_.isspace()):
            raise ValueError("The cut-off start dates cannot be None or empty")

        elif not bool(self.cutoff_start_ and not self.cutoff_start_.isspace()):
            raise ValueError("The cut-off end dates cannot be None or empty")

        elif not isinstance(self.param_grid_, dict):
            raise TypeError(
                'Make sure the "param_grid parameter" passed is of type dictionary.'
            )

        elif len(self.param_grid_) == 0:
            raise ValueError('"param_grid" parameter cannot be empty')

        elif not set(list(self.param_grid_.keys())).issubset(
            ["changepoint_prior_scale", "seasonality_prior_scale", "seasonality_mode"]
        ):
            raise ValueError(
                'Make sure the values in the "param_grid" are the ones expected by the Prophet() model '
            )

    def tune_hyperparameters(self, df: pd.DataFrame, param_grid: dict = None):
        """Function to tune the hyperparameters

        Args:
            df (pd.DataFrame): Dataframe with the data to be tuned
            param_grid (dict, optional): Parameters to be tuned. If None, defaults to the one provided in init(). Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with the optimal parameters.
        """

        cutoff_start = datetime.strptime(self.cutoff_start_, "%Y-%m-%d")
        cutoff_end = datetime.strptime(self.cuttoff_end_, "%Y-%m-%d")
        cutoffs = pd.date_range(start=cutoff_start, end=cutoff_end, freq="MS")

        if (param_grid == None) or (len(param_grid) == 0):
            # Generate all combinations of parameters
            all_params = [
                dict(zip(self.param_grid_.keys(), v))
                for v in itertools.product(*self.param_grid_.values())
            ]
        else:
            all_params = [
                dict(zip(param_grid.keys(), v))
                for v in itertools.product(*param_grid.values())
            ]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:

            m = Prophet(
                interval_width=0.95,
                growth="linear",
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                **params
            ).add_seasonality(name="yearly", period=365, fourier_order=5)

            m.fit(df)

            df_cv = cross_validation(
                model=m, horizon="90 days", cutoffs=cutoffs, parallel="processes"
            )
            df_p = performance_metrics(df_cv, rolling_window=1)
            rmses.append(df_p["rmse"].values[0])

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results["rmse"] = rmses
        tuning_results = tuning_results.sort_values("rmse")
        best_params = all_params[np.argmin(rmses)]

        return tuning_results, best_params
