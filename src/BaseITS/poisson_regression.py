from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import statsmodels.api as sm
import pandas as pd


class PoissonITS(BaseEstimator):
    """An SK-learn wrapper class to statsmodel's poisson regression.

    Args:

        fit_intercept: boolean, default= True
        fitted: boolean, parameter to indicate if intercept should be fitted.

    Methods:

        fit(...)
        predict(...)
        fit_predict(...)
        summary(...)

    """

    def __init__(self, fit_intercept=True):
        """Constructor

        Args:

            fit_intercept (bool, optional): _description_. Defaults to True.
        """
        self.fit_intercept = fit_intercept
        # self.__validate_inputs()

    # TODO Implement this
    def __validate_inputs(self):

        raise NotImplementedError("TODO")

    def fit(self, X: pd.DataFrame, y: pd.Series, offset: float = None):
        """A reference implementation of a fitting function.

        Args:
             X : {array-like, sparse matrix}, shape (n_samples, n_features)
                 The training input samples.
             y : array-like, shape (n_samples,) or (n_samples, n_outputs)
                 The target values (class labels in classification, real numbers in
                 regression).
             offset (float, optional): _description_. Defaults to None.
         Returns:
             self : PoissonITS object
                 Returns self.

        """
        if self.fit_intercept:
            X = sm.add_constant(X)

        self.X_ = X
        self.y_ = y
        self.model_ = sm.GLM(
            endog=y, exog=X, offset=offset, family=sm.families.Poisson()
        )
        self.results_ = self.model_.fit(cov_type="HC1")
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame, prediction_df: pd.DataFrame, factor: int = 1):
        """A reference implementation of a predicting function.

        Args:
            X : (array-like, sparse matrix), shape (n_samples, n_features)
                The training input samples tranformed earlier.

            prediction_df (pd.DataFrame):  Prediction dataframe before transformation.
            factor(int) {Optional}: value to standardize the data

        Returns:

            y : ndarray, shape (n_samples,)
                Returns an array of ones.

        """
        # check if model is fitted
        check_is_fitted(self, "is_fitted_")

        # check if X has correct shape
        X = check_array(X, accept_sparse=True)

        if self.fit_intercept:
            X = sm.add_constant(X)

        predictions = self.results_.get_prediction(X)
        yhat = predictions.predicted_mean
        prediction_CI = predictions.conf_int()
        yhat_lower = prediction_CI[:, 0]
        yhat_upper = prediction_CI[:, 1]
        prediction_results = pd.DataFrame(
            zip(yhat, yhat_lower, yhat_upper),
            columns=["yhat", "yhat_lower", "yhat_upper"],
        )

        forecast = pd.concat([prediction_df[["ds", "y"]], prediction_results], axis=1)

        forecast["change"] = forecast["y"] - forecast["yhat"]
        forecast["percent_change"] = forecast.apply(
            lambda row: round((row.y - row.yhat) / row.yhat * 100, 2), axis=1
        )
        forecast[["y", "yhat", "yhat_lower", "yhat_upper", "change"]] = (
            forecast[["y", "yhat", "yhat_lower", "yhat_upper", "change"]] * factor
        )

        return forecast

    def fit_predict(self):

        raise NotImplementedError(
            "This function cannot be implemented because the predictions need to be transformed mannualy by the user"
        )

    def summary(self):
        print(self.results_.summary())


# fit_transform()
