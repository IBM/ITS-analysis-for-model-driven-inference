import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd
from prophet.plot import plot_cross_validation_metric
from BaseITS.utils import *

# All plots saved in a plots folder


class Plotting:
    """Class to handle the plotting given a fixed expectation of inputs. This class is an addition, Users can implement more plotting functions to visualize their data & outputs.

    Implements:

        plot_sphaghetti: Line plot with all the provided outcomes and locations.
        plot_cumulative: Cumulative plots.
        plot_count_its: Counts plots.
        plot_percent_change: Percentage plots.
        plot_diff: Difference of values plots.
        plot_boxplots: Boxplots
        plots_metrics_distribution: Kernel Density Estimate plot

    """

    def __init__(
        self,
        intervention_end_date: str,
        forecast: dict(),
        data: pd.DataFrame,
        outcome_labels: dict() = {},
        file_path: str = "plots/",
    ):
        """Constructor

        Args:
            intervention_end_date(str):
            forecast (dict): forecast output from prophet model
            data (pd.DataFrame): dataset to be used
            outcome_labels (dict, optional): label mapping of the outcomes. Defaults to {}.
            file_path(str, optional): File path to save the plots. Defaults to 'plots/'

        """
        self.intervention_end_date = intervention_end_date
        self.forecast = forecast
        self.data = data
        self.outcome_labels = outcome_labels
        self.file_path = file_path
        self.__validate_data

    # TODO Add an implementation that allows calling the constructor/functions without the inputs
    def __validate_data(self):
        # TODO: Implement this function
        pass

    def plot_sphaghetti(self, id_var: str, x_var: str, y_var: str, title: str):

        """
        Plot sphaghetti

        @params
        data: data in wide format (dataframe)
        id_var: column name of unique ids of individuals (str) i.e Regions, Districts
        x_var: name of x variable (str) i.e Date column name
        y_var: name of y variable (str) i.e outcome measure to be plotted column name i.e Diabetes
        """

        sns.set(rc={"figure.figsize": (12, 7)})
        mpl.rcParams["font.size"] = 8.0
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"

        ids = self.data[id_var]

        for i in ids:
            df = self.data[[x_var, y_var]][self.data[id_var] == i]
            df[y_var] = df[y_var].fillna(df[y_var].mean())
            # z = np.abs(stats.zscore(df[y_var]))
            # df = df[z<3]
            plt.plot(
                df[x_var], df[y_var], marker="", color="black", linewidth=0.1, alpha=0.1
            )

        plt.ylabel("Count", fontsize=18, fontweight="bold")
        # plt.xlabel(x_var, fontsize = 14, fontweight='bold')

        plt.title(title, fontsize=20, fontweight="bold")
        plt.xticks(rotation=90, fontsize=18, fontweight="bold")
        plt.yticks(fontsize=18, fontweight="bold")

        save_plots(self.file_path + title)
        plt.xlabel("Date")
        plt.show()
        plt.close()

    def __plot_all_cumulative_its(self):
        return NotImplementedError(
            "User to implement this themselves. Exposing the function only for plotting one outcome, location, intervention, model only."
        )

    def plot_cumulative(
        self,
        forecast: pd.DataFrame(),
        unit: str,
        outcome: str,
        prediction_start_date: str,
        prediction_end_date: str,
        normalise: bool = False,
    ):
        """
        Function to plot the cumulative forecast.

        @params:
        unit: Geographical/Organizational unit (string)
        outcome: Outcome measure (string)
        forecast: Dataframe with the prophet forecast output for the geographical/organizational unit and outcome (pd.DataFrame)
        prediction_start_date: Prediction start date (datetime)
        prediction_end_date: Prediction end date (datetime)
        normalise: Data normalised (boolean)

        """

        sns.set(rc={"figure.figsize": (10, 8)})
        sns.set_style("white")
        mpl.rcParams["font.size"] = 8.0
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"

        prediction_start_date = datetime.strptime(prediction_start_date, "%Y-%m-%d")
        prediction_end_date = datetime.strptime(prediction_end_date, "%Y-%m-%d")
        plt.xlim([forecast["ds"].min(), prediction_end_date])
        plt.axvspan(
            xmin=prediction_start_date,
            xmax=prediction_end_date,
            color="grey",
            alpha=0.2,
            lw=0,
        )
        plt.plot(
            forecast["ds"],
            forecast["y"].cumsum(),
            color="orange",
            label="actual values",
        )
        plt.plot(
            forecast["ds"],
            forecast["yhat"].cumsum(),
            color="#33adff",
            label="predicted values",
        )

        plt.xticks(rotation=90, fontsize=18, fontweight="bold")
        plt.yticks(fontsize=18, fontweight="bold")

        plt.xlabel("")

        if normalise:
            plt.ylabel("Cumulative counts per 100k", fontsize=18, fontweight="bold")
            title = "{} in {} Region normalised".format(
                self.outcome_labels[outcome], unit
            )
        else:
            plt.ylabel("Cumulative counts", fontsize=18, fontweight="bold")
            title = "{} in {} ".format(self.outcome_labels[outcome], unit)

        plt.title(title, fontsize=20, fontweight="bold")
        plt.legend(loc="upper left", fontsize=12)
        save_plots(file_path="plots/", plot_name=title)

        plt.show()
        plt.close()

    def __plot_all_counts_its(self):
        return NotImplementedError(
            "User to implement this themselves. Exposing the function only for plotting one outcome, location, intervention, model only."
        )

    def plot_count_its(
        self,
        forecast: pd.DataFrame(),
        unit: str,
        outcome: str,
        prediction_start_date: str,
        prediction_end_date: str,
        normalise: bool = False,
    ):
        """
        Function to plot the forecast counts.

        @params:
        unit: Geographical unit (string)
        outcome: Outcome measure (string)
        forecast: Dataframe with the prophet forecast output for the geographical unit and outcome(pd.DataFrame)
        prediction_start_date: Prediction start date (datetime)
        prediction_end_date: Prediction end date (datetime)
        normalise: Data normalised (boolean)

        """
        sns.set(rc={"figure.figsize": (10, 8)})
        sns.set_style("white")
        mpl.rcParams["font.size"] = 8.0
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"

        prediction_start_date = datetime.strptime(prediction_start_date, "%Y-%m-%d")
        prediction_end_date = datetime.strptime(prediction_end_date, "%Y-%m-%d")

        plt.axvspan(
            xmin=prediction_start_date,
            xmax=prediction_end_date,
            color="grey",
            alpha=0.2,
            lw=0,
        )

        plt.scatter(
            forecast["ds"],
            forecast["y"],
            facecolors="none",
            edgecolors="black",
            s=20,
            label="observed values",
        )
        plt.xlim([forecast["ds"].min(), prediction_end_date])

        plt.plot(
            forecast["ds"], forecast["yhat"], color="#33adff", label="predicted values"
        )
        plt.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="#33adff",
            alpha=0.25,
        )

        plt.plot(
            forecast["ds"],
            forecast["trend"],
            color="red",
            linestyle="--",
            label="predicted trend",
        )
        plt.fill_between(
            forecast["ds"],
            forecast["trend_lower"],
            forecast["trend_upper"],
            color="red",
            alpha=0.2,
        )

        plt.xticks(rotation=90, fontsize=18, fontweight="bold")
        plt.yticks(fontsize=18, fontweight="bold")

        plt.xlabel("")

        if normalise:
            plt.ylabel("count per 100k", fontsize=18, fontweight="bold")
            title = "{} in {} normalised".format(self.outcome_labels[outcome], unit)
        else:
            plt.ylabel("counts", fontsize=18, fontweight="bold")
            title = "{} in {} ".format(self.outcome_labels[outcome], unit)

        plt.title(title, fontsize=20, fontweight="bold")
        plt.legend(loc="upper left", fontsize=12)
        save_plots(file_path="plots/", plot_name=title)
        plt.show()
        plt.close()

    def plot_percent_change(self, dataset: pd.DataFrame, prediction_start_date: str):

        """
        Function to plot the percentage change of the outcomes and the predicted values

        @params:
        data: Dataframe with the selected metrics from the forecast (pd.DataFrame)
        prediction_start_date: Prediction start date (str)

        """
        df = dataset[dataset["ds"] == prediction_start_date].copy()
        df = df.sort_values(["outcome", "mean_percent_change_after"], ascending=False)

        prediction_start_date = datetime.strptime(prediction_start_date, "%Y-%m-%d")

        sns.set(rc={"figure.figsize": (10, 9)})
        mpl.rcParams["font.size"] = 12.0
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"

        spot = 0
        for outcome in set(df["outcome"]):
            spot += 1
            plt.subplot(3, 1, spot)

            x = df["location"][df["outcome"] == outcome]
            y = np.array(
                list(df["mean_percent_change_after"][df["outcome"] == outcome])
            )
            conf = np.array(
                list(df["percent_change_conf_int_after"][df["outcome"] == outcome])
            )
            yconf = np.c_[y - conf[:, 0], conf[:, 1] - y].T

            plt.plot(y, x, ".", color="blue")
            plt.axvline(x=0, linewidth=1, linestyle="--", color="grey")
            title = "{}".format(self.outcome_labels[outcome])
            plt.title(title, loc="center", fontsize=12, fontweight="bold")
            plt.xlabel(
                "percent change in {} counts".format(self.outcome_labels[outcome]),
                fontsize=12,
                fontweight="bold",
            )
            plt.errorbar(y, x, xerr=yconf, fmt=" ", color="blue")
        suptitle = "Mean and 95% CI of Percent Change in Predicted vs Actual Counts ({} {} to December 2020)".format(
            prediction_start_date.strftime("%B"), prediction_start_date.year
        )
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")
        plt.tight_layout(pad=4.0)
        save_plots(self.file_path, "{}".format(suptitle))
        # plt.savefig("plots/{}.png".format(suptitle), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_diff(self, dataset: pd.DataFrame, prediction_start_date: str):
        """
        Function to plot the difference of the outcomes and the predicted values

        @params:
        data: Dataframe with the metrics from the forecast (pd.DataFrame)
        prediction_start_date: Prediction start date (str)

        """

        df = dataset[dataset["ds"] == prediction_start_date].copy()
        df = df.sort_values(["outcome", "mean_change_after"], ascending=False)

        prediction_start_date = datetime.strptime(prediction_start_date, "%Y-%m-%d")

        sns.set(rc={"figure.figsize": (11, 9)})
        mpl.rcParams["font.size"] = 12.0
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"

        spot = 0
        for outcome in set(df["outcome"]):
            spot += 1
            plt.subplot(3, 1, spot)

            x = df["location"][df["outcome"] == outcome]
            y = np.array(list(df["mean_change_after"][df["outcome"] == outcome]))
            conf = np.array(list(df["change_conf_int_after"][df["outcome"] == outcome]))
            yconf = np.c_[y - conf[:, 0], conf[:, 1] - y].T

            plt.plot(y, x, ".", color="blue")
            plt.axvline(x=0, linewidth=1, linestyle="--", color="grey")
            title = "{}".format(self.outcome_labels[outcome])
            plt.title(title, loc="center", fontsize=12, fontweight="bold")
            plt.xlabel(
                "mean of the difference in predicted vs actual {} counts".format(
                    self.outcome_labels[outcome]
                ),
                fontsize=12,
                fontweight="bold",
            )
            plt.errorbar(y, x, xerr=yconf, fmt=" ", color="blue")
        suptitle = "Mean and 95% CI of Difference in Predicted vs Actual Counts ({} {} to December 2020)".format(
            prediction_start_date.strftime("%B"), prediction_start_date.year
        )
        plt.suptitle(suptitle, fontsize=14, fontweight="bold")
        plt.tight_layout(pad=4.0)
        save_plots(self.file_path, "{}".format(suptitle))
        # plt.savefig("plots/{}.png".format(suptitle), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_boxplots(self, x: str, y: str, title: str):
        """Function to plot boxplots

        Args:
            x (str): date column
            y (str): data value column to be plotted
            title (str): title of plot
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.boxplot(
            self.data[x].dt.strftime("%Y-%m"), self.data[y], ax=ax, showfliers=False
        )
        # sns.violinplot(data[x].dt.strftime('%Y-%m'), data[y], ax=ax,cut=0)
        plt.yticks(fontsize=14, fontweight="bold")
        plt.xticks(rotation="vertical")
        plt.xticks(rotation=90, fontsize=14, fontweight="bold")
        plt.ylabel("Count", fontsize=14, fontweight="bold")
        plt.xlabel("")
        plt.title(title, fontsize=20, fontweight="bold")
        save_plots(self.file_path, title + "time_series_boxplots.png")
        # plt.savefig(
        #     self.file_path + title + "time_series_boxplots.png", dpi=300, bbox_inches="tight"
        # )
        plt.show()
        plt.close()

    def plots_metrics_distribution(
        self, prediction_start_date: str, prediction_end_date: str
    ):

        """Function to plot metrics distribution before and after interruption

         @params:
        prediction_start_date: Prediction start date (str)
        prediction_end_date: Prediction end date (str)


        """

        prediction_start_date = datetime.strptime(prediction_start_date, "%Y-%m-%d")
        prediction_end_date = datetime.strptime(prediction_end_date, "%Y-%m-%d")

        df_before = self.forecast[self.forecast["ds"] < prediction_start_date]

        df_after = self.forecast[
            (self.forecast["ds"] >= prediction_start_date)
            & (self.forecast["ds"] <= prediction_end_date)
        ]
        sns.set(rc={"figure.figsize": (8, 6)})
        sns.set_style("white")
        mpl.rcParams["font.size"] = 8.0
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"
        sns.kdeplot(df_before["yhat-y"], shade=True, label="Before")
        sns.kdeplot(df_after["yhat-y"], shade=True, label="After")
        title = "Distribution of yhat-y before and after interruption"
        plt.title(
            title,
            fontsize=16,
            fontweight="bold",
        )
        plt.legend()
        save_plots(self.file_path, title)
        plt.show()
        plt.close()

    def plot_cv_metric(
        self,
        location: str,
        outcome: str,
        df_cv: pd.DataFrame,
        prediction_start_date: str,
        outcome_labels: dict,
    ):

        """
        Function to plot the cross-validation metrics.

        @params:
        location: Geographical unit (string)
        outcome: Outcome measure (string)
        df_cv: Dataframe with date and the outcome value for the geographical unit and outcome. (pd.DataFrame)
        seasonality_mode: tuned prophet model parameter (string)
        changepoint_prior_scale: tuned prophet model parameter (double)
        seasonality_prior_scale: tuned prophet model parameter (double)
        cutoff_start: Prediction start date (datetime)

        """
        sns.set(rc={"figure.figsize": (10, 8)})
        sns.set_style("white")
        mpl.rcParams["font.size"] = 8.0
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"

        plot_cross_validation_metric(df_cv, metric="mape")
        plt.xticks(fontsize=18, fontweight="bold")
        plt.yticks(fontsize=18, fontweight="bold")
        plt.xlabel("Horizon", fontsize=18, fontweight="bold")
        plt.ylabel("MAPE", fontsize=18, fontweight="bold")

        prediction_start_date = datetime.strptime(prediction_start_date, "%Y-%m-%d")
        title = "MAPE for {} in {} ({} {})".format(
            outcome_labels[outcome],
            location,
            prediction_start_date.strftime("%B"),
            prediction_start_date.year,
        )

        plt.title(title, fontsize=20, fontweight="bold")
        plt.savefig("plots/{}.png".format(title), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()
