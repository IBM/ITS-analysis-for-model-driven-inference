import pandas as pd
from prophet import Prophet
from sklearn.utils.validation import check_is_fitted


class ProphetITS:
    """A wrapper class that uses Prophet https://pypi.org/project/prophet/ to forecast"""

    def __init__(self, df=None):
        """Default Constructor

        Args:
            df (pd.DataFrame, optional): dataFrame with the timeseries data. Defaults to None.
            __validate_inputs(): function that validates the inputs provided.

        Methods:
            fit(df,seasonality_mode: str = "additive",seasonality_prior_scale:float = 0.5,changepoint_prior_scale: float = 0.5)
            predict(df)
            fit_predict(df,seasonality_mode: str = "additive",seasonality_prior_scale:float = 0.5,changepoint_prior_scale: float = 0.5)
            summary()

        """
        self.df = df
        self.__validate_inputs

    def __validate_inputs():
        raise NotImplementedError("TODO")

    def fit(
        self,
        df: pd.DataFrame,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float = 0.5,
        changepoint_prior_scale: float = 0.5,
    ):
        """Fit function of the wrapper Prophet class

        Args:
            df (pd.DataFrame): DataFrame with the training data.
            seasonality_mode (str, optional): Seasonality mode experienced by the data . Defaults to "additive".
            seasonality_prior_scale (float, optional): seasonality prior scale  that has been optimised for the dataset. Defaults to 0.5.
            changepoint_prior_scale (float, optional): changepoint prior scale that has been optimised for the dataset. Defaults to 0.5.

        Returns:
            self(ProphetITS): instance of the class that is fitted using the prophet model.
        """

        self.df_training_ = df

        self.model_ = Prophet(
            interval_width=0.95,
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        ).add_seasonality(name="yearly", period=365, fourier_order=5)

        self.results_ = self.model_.fit(self.df_training_)
        self.is_fitted_ = True

        return self.results_

    def predict(self, df: pd.DataFrame):
        """A reference implementation of a predicting function.
        Parameters
        ----------
        df : (pd.DataFrame) DataFrame with the testing data. Both X and y
        Returns
        -------
        forecast : (pd.DataFrame) Returns a dataframe with forecast
        """
        self.df_testing_ = df

        # get dates to forecast

        future = self.df_testing_[["ds"]].copy()

        forecast = self.results_.predict(future)

        # forecast["y"] = self.df_testing_["y"]
        forecast["y"] = [i for i in self.df_testing_["y"]]
        forecast["change"] = forecast["y"] - forecast["yhat"]
        forecast["percent_change"] = forecast.apply(
            lambda row: round((row.y - row.yhat) / row.yhat * 100, 2), axis=1
        )

        return forecast

    def fit_predict(
        self,
        df: pd.DataFrame,
        seasonality_mode="additive",
        seasonality_prior_scale=0.5,
        changepoint_prior_scale=0.5,
    ):
        # note: uses same df for training and testing

        self.df_training_ = df

        self.model_ = Prophet(
            interval_width=0.95,
            growth="linear",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        ).add_seasonality(name="yearly", period=365, fourier_order=5)

        self.results_ = self.model_.fit(self.df_training_)

        future = self.df_training_[["ds"]].copy()

        forecast = self.results_.predict(future)

        forecast["y"] = [i for i in self.df_training_["y"]]
        forecast["change"] = forecast["y"] - forecast["yhat"]
        forecast["percent_change"] = forecast.apply(
            lambda row: round((row.y - row.yhat) / row.yhat * 100, 2), axis=1
        )

        # model_fitted = self.fit(
        #     df, seasonality_mode, seasonality_prior_scale, changepoint_prior_scale
        # )

        # predictions = model_fitted.predict(df)
        # print("In FIT_PREDICT")
        # print(predictions.columns)

        return forecast

    def summary(self):
        return "Implementation coming soon. Not implemented in the base prophet class used."
