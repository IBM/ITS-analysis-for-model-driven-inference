import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomTransform(BaseEstimator, TransformerMixin):
    """Class to transform dataframe for Poisson Regression by adding harmonic.

    Args:
        BaseEstimator (Sklearn): Base class for all estimators in scikit-learn.
        TransformerMixin (Sklearn): Mixin class for all transformers in scikit-learn.
    """

    def __init__(self, columns: list):
        """Constructor

        Args:
            columns (list): list of columns to be transformed
        """
        self.columns = columns

    def __init__(
        self,
        columns: list,
        seasonally_adjusted: bool = True,
        var_name: str = "month",
        nfreq: int = 2,
        period: int = 12,
        fit_intercept: bool = False,
    ):
        """Constructor

        Args:
            columns (list): list of columns to be transformed
            seasonally_adjusted (bool, optional): Boolean value whether to adjust for seasons. Defaults to True.
            var_name (str, optional): Column name for the time/month column. Defaults to "month".
            nfreq (int, optional): Frequency. Defaults to 2.
            period (int, optional): Period duration. Defaults to 12.
            fit_intercept (bool, optional): Boolean value whether to perform fit_intercept. Defaults to False.
        """
        self.columns = columns
        self.seasonally_adjusted = seasonally_adjusted
        self.var_name = var_name
        self.nfreq = nfreq
        self.period = period
        self.fit_intercept = fit_intercept

    # TODO: Implement validate_inputs()

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Function to fit the model

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None):
        """Function to transform the variables

        Args:
            X (pd.DataFrame): Dataframe with the harmonic inputs
            y (pd.Series, optional): Series of the outcome variable. Defaults to None.

        Returns:
            pd.DataFrame: _description_
        """
        if self.seasonally_adjusted:
            if self.var_name not in X.columns:
                print("Harmonic variable not found in the data")
                return
            if not self.nfreq > 0:
                print("nfreq > 0 is not true")
                return
            v = X[self.var_name]
            if not all(isinstance(i, (int, float)) for i in v):
                print("All values must be numeric")
                return

            N = list(range(0, self.nfreq))
            k = [(2**i) * 2 * np.pi / self.period for i in N]
            M = np.outer(v, k)
            simM = np.sin(M)
            cosM = np.cos(M)

            if self.fit_intercept:
                a = np.array([1] * len(cosM))
                harmonic_X = pd.DataFrame(np.column_stack([a, simM, cosM]))
                harmonic_X.columns = [
                    "harmonic({},{},{}) intercept".format(
                        self.var_name, self.nfreq, self.period
                    )
                ] + [
                    "harmonic({},{},{}) {}".format(
                        self.var_name, self.nfreq, self.period, i + 1
                    )
                    for i in harmonic_X.columns[1:]
                ]
            else:
                harmonic_X = pd.DataFrame(np.column_stack([simM, cosM]))
                harmonic_X.columns = [
                    "harmonic({},{},{}) {}".format(
                        self.var_name, self.nfreq, self.period, i + 1
                    )
                    for i in harmonic_X.columns
                ]

            return pd.concat([X[self.columns], harmonic_X], axis=1)
        else:
            return X[self.columns]
