import pandas as pd
from datetime import datetime

from multiprocessing.pool import Pool
from multiprocessing import cpu_count

from .utils import extract_inputs

from .poisson_regression import PoissonITS
from .prophet_model import ProphetITS
from .pre_processing import align_prophet_naming_convection

# BaseITS
class BaseITS:
    """This class assumes that the data is already pre-processed to the required format.
    If not pre-processed, the user should use the functions in pre-processing & custom_transform to pre-process their data
    Class supports wide-format datasets only. Refer to readme on how to structure your dataset.

       Args:
           outcome (list, optional): List of the outcome labels. Defaults to None.
           location (list, optional): List of the location labels. Defaults to None.
           interruption_date (list, optional): List of the interruption dates. Defaults to None.
           model (list, optional): List of the models labels. Defaults to None.
           verbose (bool, optional): Boolean variable to log outputs. Defaults to False.

       Methods:
           fit(df: pd.DataFrame, X: pd.Series, y: pd.Series, offset: float = None)
           predict( df: pd.DataFrame, X: pd.Series, y: pd.Series, offset: float = None)
           fit_predict(df: pd.DataFrame, X: pd.Series, y: pd.Series)


    """

    def __init__(
        self,
        outcome: list = None,
        location: list = None,
        interruption_date: list = None,
        model: list = None,
        verbose: bool = False,
    ):

        self.outcome_ = outcome
        self.location_ = location
        self.interruption_date_ = interruption_date
        self.model_ = model
        self.verbose = verbose
        self.prophet_model_ = ProphetITS()
        self.poisson_regression_ = PoissonITS()
        self.__validate_inputs_()

        # Start implementation with assumption we only accept wide datasets. User expected to convert their data to align to this.

        # Write a validation to make sure inputs are as lists.

    def __validate_inputs_(self):
        """Class to validate that the inputs provided are the correct ones

        Returns:
            ValueError: Raises ValueErrors based on the implemented checks
        """

        NoneType = type(None)
        # print(type(self.location_), isinstance(self.location_, NoneType))

        if isinstance(self.location_, NoneType):
            raise ValueError(
                'Location list not provided. Supply the "location" to the class declaration'
            )
        if isinstance(self.outcome_, NoneType):
            raise ValueError(
                'Outcome list not provided. Supply the "location" to the class declaration'
            )
        if isinstance(self.interruption_date_, NoneType):
            raise ValueError(
                'Interruption dates list not provided. Supply the "location" to the class declaration'
            )
        if not isinstance(self.location_, list):
            return TypeError('Parameter "location" should of type "list".')

        if not isinstance(self.outcome_, list):
            return TypeError('Parameter "outcome" should of type "list".')

        if not isinstance(self.interruption_date_, list):
            return TypeError('Parameter "interruption_date_" should of type "list".')

        if not isinstance(self.model_, list):
            return TypeError('Parameter "locamodeltion" should of type "list".')

        if self.model_ not in [["prophet"], ["poisson"], ["poisson", "prophet"]]:
            raise ValueError('Parameter "model" should be "prophet" or "poisson".')

        if len(self.outcome_) is 0:
            raise ValueError(
                'Outcome list is empty. Supply "outcomes" to the class declaration'
            )

        if (len(self.location_)) is 0:
            raise ValueError(
                'Location list is empty. Supply the "location" to the class declaration'
            )
        if len(self.interruption_date_) is 0:
            raise ValueError(
                'Interruption dates list is empty. Supply the "Interruption dates" to the class declaration'
            )

    def __validate_fit_predict(self, df: pd.DataFrame):
        """Function to validate fit & prediction specific values conform to the expected values. Not done in _init_ as user is not required to pass a dataframe.

        Args:
            df (pd.DataFrame): DataFrame with the data

        Raises:
            ValueError: Raises errors based on the Raises ValueErrors based on the implemented checks

        """
        # confirm that the provided list of outcomes & locations is in df
        if not set(self.location_).issubset(df["location"].unique().tolist()):
            raise ValueError(
                'DataFrame must have the unique values(rows) provided in the "location" parameter'
            )

        if not set(self.outcome_) <= set(df.columns.tolist()):
            raise ValueError(
                'DataFrame must have the columns provided in the "outcome" parameter'
            )

        if "location" not in df.columns:
            raise ValueError(
                'Dataframe must have column "location" with the ' "location."
            )

    def fit(self, df: pd.DataFrame, X: pd.Series, y: pd.Series, offset: float = None):
        """Function called by user to fit their models.(prophet or poisson)

        Args:
            df (pd.DataFrame): DataFrame with the data
            offset (float, optional): _description_. Defaults to None.

        Returns:
            BaseITS: Fitted object of the class.
        """

        self.__validate_inputs_()
        self.__validate_fit_predict(df)
        len_outcomes = len(self.outcome_)
        len_locations = len(self.location_)

        if (len_outcomes > 1) or (len_locations > 1) or (len(self.model_) > 1):
            # works for multiple locations and outcomes

            # call the pool fit method for each location and outcome
            fitted_model_results = self.__pool_fit(
                df=df, offset=offset, model=self.model_
            )
            return fitted_model_results

        else:
            fitted_model = self.__fit_once(X=X, y=y, offset=offset, model=self.model_)
            return fitted_model

    def __fit_once(self, X: pd.Series, y: pd.Series, offset: float = None, model=None):

        """Private class where the fit occurs to only one occurance of either model, outcome, location, intervention_date provided.

        Raises:
            ValueError: Raises ValueError incase the inputs are not as expected

        Returns:
            model: Fitted model
        """
        if model == "prophet" or ["prophet"]:
            if self.verbose:
                print("prophet fit once")
            concat_df = pd.concat([X, y], axis=1)
            concat_df = align_prophet_naming_convection(
                concat_df, date_col_name=X.name, y_col_name=y.name
            )
            self.prophet_model_.fit(concat_df)
            return self.prophet_model_

        elif model == "poisson" or ["poisson"]:
            if offset == None:
                if self.verbose:
                    print("poisson fit once error")
                raise ValueError("Offset needs to be defined for poisson regression")
            else:
                if self.verbose:
                    print("poisson fit once ")
                self.poisson_regression_.fit(X, y, offset)
                return self.poisson_regression_

        else:
            if self.verbose:
                print("fit once error")
            raise ValueError("Specify either 'prophet' or 'poisson' models")

    def __pool_fit(
        self,
        df: pd.DataFrame,
        offset: float = None,
        num_threads: int = cpu_count(),
        model=None,
    ):
        """Private function to fit more than one instance of outcomes, locations, models, intervention_dates

        Args:
            df (pd.DataFrame): DataFrame with the data
            offset (float, optional): _description_. Defaults to None.
            num_threads (int, optional): Number of cpu counts that will allow multi-threading. Defaults to cpu_count().

        Returns:
            model: fitted model
        """
        if self.verbose:
            print(num_threads, "num_threads")
        if num_threads > 1:
            # define thread pool
            pool = Pool(processes=num_threads)

            # send jobs to thread pool
            results = dict()
            for location in self.location_:
                results[location] = dict()
                # location_results = results.get(location, {})
                df_copy = df[df["location"] == location].copy().reset_index(drop=True)

                for outcome in self.outcome_:
                    results[location][outcome] = dict()
                    # outcome_results = location_results.get(outcome, {})
                    df_copy["y"] = df_copy[outcome].copy()

                    for intervention in self.interruption_date_:
                        results[location][outcome][intervention] = dict()
                        # intervention_results = outcome_results.get(intervention, {})

                        prediction_start_date = datetime.strptime(
                            intervention, "%Y-%m-%d"
                        )
                        modeling_df = df_copy[
                            df_copy["ds"] <= prediction_start_date
                        ].copy()

                        for model in self.model_:
                            results[location][outcome][intervention][model] = dict()
                            # model_results = intervention_results.get(model, {})
                            # model_results[model] = dict()
                            if self.verbose:
                                print(
                                    "{} - {} - {} - {}".format(
                                        location, outcome, intervention, model
                                    )
                                )

                            fitted_model = self.__fit_once(
                                X=modeling_df["ds"],
                                y=modeling_df["y"],
                                offset=offset,
                                model=model,
                            )

                            results[location][outcome][intervention][model][
                                "fitted_model"
                            ] = fitted_model
        return results

    def predict(
        self, df: pd.DataFrame, X: pd.Series, y: pd.Series, offset: float = None
    ):
        """Function used to forecast using the previously fitted models

        Args:
            df (pd.DataFrame): Dataset with the data to be used
            offset (float, optional): _description_. Defaults to None.
        """
        self.__validate_inputs_()
        self.__validate_fit_predict(df)
        len_outcomes = len(self.outcome_)
        len_locations = len(self.location_)

        if self.verbose:
            print(len_outcomes, len_locations)

        if (len_outcomes > 1) or (len_locations > 1) or (len(self.model_) > 1):
            # call the pool fit method for each location and outcome
            predictions = self.__pool_predict(df=df, offset=offset)
            return predictions

        else:
            predictions = self.__predict_once(
                X=X, y=y, offset=offset, model=self.model_
            )
            return predictions

    def __predict_once(
        self,
        X: pd.Series,
        y: pd.Series = None,
        model: str = "prophet",
        offset: float = None,
    ):
        """Private function to predict only one occurence of outcome, location, model, intervention

        Args:
            X (pd.Series): Series with the X (ds) column
            y (pd.Series, optional): Series with the y column. Defaults to None.

        Raises:
            ValueError: Raises ValueError incase the inputs are not as expected

        Returns:
            pd.DataFrame: dataframe with the forecast results
        """

        if model == "prophet" or ["prophet"]:
            concat_df = pd.concat([X, y], axis=1)
            concat_df = align_prophet_naming_convection(
                concat_df, date_col_name=X.name, y_col_name=y.name
            )
            predictions = self.prophet_model_.predict(concat_df)
            return predictions

        elif model == "poisson" or ["poisson"]:
            if y == None:
                raise ValueError("y series needs to be defined for poisson regression")
            else:
                predictions = self.poisson_regression_.predict(X, y)
                return predictions

        else:
            raise ValueError("Specify either prophet or the poisson models")

    def __pool_predict(
        self,
        df: pd.DataFrame,
        offset: float = None,
        num_threads: int = cpu_count(),
    ):
        """Private function to fit more than one instance of outcomes, locations, models, intervention_dates

        Args:
            df (pd.DataFrame): Dataset with the data
            offset (float, optional): _description_. Defaults to None.
            num_threads (int, optional): cpu_count(). Defaults to cpu_count().

        Returns:
            dict: Dictionary with results of the predictions of the provided instance of outcomes, locations, models, intervention_dates
        """
        if self.verbose:
            print(num_threads, "num_threads")
        if num_threads > 1:
            # define thread pool
            pool = Pool(processes=num_threads)

            # send jobs to thread pool
            results = dict()
            for location in self.location_:
                results[location] = dict()
                # location_results = results.get(location, {})
                df_copy = df[df["location"] == location].copy().reset_index(drop=True)

                for outcome in self.outcome_:
                    results[location][outcome] = dict()
                    # outcome_results = location_results.get(outcome, {})
                    df_copy["y"] = df_copy[outcome].copy()

                    for intervention in self.interruption_date_:
                        results[location][outcome][intervention] = dict()
                        # intervention_results = outcome_results.get(intervention, {})

                        prediction_start_date = datetime.strptime(
                            intervention, "%Y-%m-%d"
                        )
                        modeling_df = df_copy[
                            df_copy["ds"] <= prediction_start_date
                        ].copy()

                        for model in self.model_:
                            results[location][outcome][intervention][model] = dict()
                            # model_results = intervention_results.get(model, {})
                            # model_results[model] = dict()
                            if self.verbose:
                                print(
                                    "{} - {} - {} - {}".format(
                                        location, outcome, intervention, model
                                    )
                                )

                            forecast_results = self.__predict_once(
                                X=modeling_df["ds"],
                                y=modeling_df["y"],
                                # TODO pass the X one only for prediction, or what the function expects. : Done
                                offset=offset,
                                model=model,  # TODO need to pass this along to other functions: Done
                            )
                            results[location][outcome][intervention][model][
                                "forecast"
                            ] = forecast_results

        return results

    def summary(self):
        """Function to generate the summary of the models

        Returns:
            _type_: summary returned by the model
        """
        if self.model_ == "prophet":
            summary = self.prophet_model_.summary()
            return summary

        elif self.model_ == "poisson":
            summary = self.poisson_regression_.summary()
            return summary

        else:
            return "Specify either prophet or poisson models"

    def fit_predict(self, df: pd.DataFrame, X: pd.Series, y: pd.Series):
        """Function to simultaneously fit and predict a function using the prophet model.
        This function does not work for poisson-regression as the data needs to be pre-processed using the custom-transform class.
        TODO Implement this fuction to check if user has already preprocessed the poisson data.

        Args:
            df (pd.DataFrame): Dataset

        Raises:
            NotImplementedError: Raises this error if user tries to use the poisson regression model
            ValueError: Raises an error if the user provided model not in this list[prophet, poisson]

        Returns:
            pd.DataFrame: DataFrame with the forecasted results.
        """
        self.__validate_inputs_()
        self.__validate_fit_predict(df)

        if self.model_ == "poisson" or self.model_ == ["poisson"]:
            raise NotImplementedError(
                "This function cannot be implemented because the predictions need to be transformed mannualy by the user"
            )
        elif self.model_ == "prophet" or self.model_ == ["prophet"]:
            len_outcomes = len(self.outcome_)
            len_locations = len(self.location_)

            if (len_outcomes > 1) or (len_locations > 1) or (len(self.model_) > 1):

                # call fit_predict from the prophet model using the loops
                return self.pool_fit_predict(df=df)
            else:
                return self.__fit_predict_once(X=X, y=y, model=self.model_)
        else:
            raise ValueError("Make sure you input the correct model name.")

    def __fit_predict_once(
        self,
        X: pd.Series,
        model: str,
        y: pd.Series = None,
    ):
        """Private function to fit and predict only one occurence of outcome, location, model, intervention

        Args:
            X (pd.Series): Series with the X (ds) column
            y (pd.Series, optional): Series with the y column. Defaults to None.

        Raises:
            ValueError: Raises ValueError incase the inputs are not as expected

        Returns:
            pd.DataFrame: dataframe with the forecast results
        """

        if model == "prophet" or ["prophet"]:
            concat_df = pd.concat([X, y], axis=1)
            concat_df = align_prophet_naming_convection(
                concat_df, date_col_name=X.name, y_col_name=y.name
            )
            predictions = self.prophet_model_.fit_predict(concat_df)
            return predictions

        else:
            raise ValueError("Specify either the 'prophet' models")

    def pool_fit_predict(
        self,
        df: pd.DataFrame,
        num_threads: int = cpu_count(),
    ):
        """Private function to fit and predictmore than one instance of outcomes, locations, models, intervention_dates

        Args:
            num_threads (cpu_count): cpu_count
            df (pd.DataFrame): Dataset

        Returns:
            pd.DataFrame: DataFrame with the forecast results
        """
        if self.verbose:
            print(num_threads, "num_threads")
        if num_threads > 1:
            # define thread pool
            pool = Pool(processes=num_threads)

            # send jobs to thread pool
            results = dict()
            for location in self.location_:
                results[location] = dict()
                # location_results = results.get(location, {})
                df_copy = df[df["location"] == location].copy().reset_index(drop=True)

                for outcome in self.outcome_:
                    results[location][outcome] = dict()
                    # outcome_results = location_results.get(outcome, {})
                    df_copy["y"] = df_copy[outcome].copy()

                    for intervention in self.interruption_date_:
                        results[location][outcome][intervention] = dict()
                        # intervention_results = outcome_results.get(intervention, {})

                        prediction_start_date = datetime.strptime(
                            intervention, "%Y-%m-%d"
                        )
                        modeling_df = df_copy[
                            df_copy["ds"] <= prediction_start_date
                        ].copy()

                        for model in self.model_:
                            results[location][outcome][intervention][model] = dict()
                            # model_results = intervention_results.get(model, {})
                            # model_results[model] = dict()
                            if self.verbose:
                                print(
                                    "{} - {} - {} - {}".format(
                                        location, outcome, intervention, model
                                    )
                                )

                            forecast_results = self.__fit_predict_once(
                                X=modeling_df["ds"],
                                y=modeling_df["y"],
                                model=model,
                            )

                            # model_results[model]["model"] = fitted_model
                            results[location][outcome][intervention][model][
                                "forecast"
                            ] = forecast_results

        # TODO Work on pool elif statement here & pool_fit(), pool_predict()
        return results


# Set call with William / Sekou / Charles to setup the pool help call
