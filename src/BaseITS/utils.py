import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# from prophet.plot import plot_cross_validation_metric
from datetime import datetime


# def get_unique_values(value):
#     return value.unique().tolist()


def check_dataset_format(df: pd.DataFrame, outcomes: list, col_name: str = "trial"):
    """Function to check dataset format. Not used as of now. Only long dataframe accepted in V1

    Args:
        df (pd.DataFrame): DataFrame with the data
        outcomes (list): outcomes to be used
        col_name (str, optional): column name provided in the init. Defaults to "trial".

    Raises:
        Exception: Excepetion rasied if outcomes not in the dataframe

    Returns:
        str: dataset_format (long,wide)
    """

    dataset_format = ""
    values = df.columns.unique().tolist()
    # print(values, "values")
    # print(any(elem in outcomes for elem in values))
    try:
        if col_name == "trial" and any(elem in outcomes for elem in values):
            dataset_format = "wide"
            return dataset_format

        elif col_name != "trial" and outcomes in df[col_name].unique().tolist():
            dataset_format = "long"
            print(dataset_format, "dataset_format")
            return dataset_format
    except Exception as e:
        dataset_format = "error"
        raise Exception("Make sure the outcomes provided are in the dataset.")


def save_plots(file_path: str, plot_name: str):
    """Function to save plots

    Args:
        file_path (str): File path
        plot_name (str): name of the plot
    """

    if os.path.dirname(file_path) != "" and not os.path.exists(
        os.path.dirname(file_path)
    ):
        os.makedirs(os.path.dirname(file_path))

    plt.savefig(file_path + plot_name + "_its.png", dpi=300, bbox_inches="tight")
    return


def extract_inputs(data: dict):
    """Function to extract inputs

    Args:
        data (dict): Dictionary with the looped results.

    Returns:
        tuple: tuple with the keys of the dictionary
    """
    for intervention_date in data.keys():
        for location in data[intervention_date].keys():
            for outcome in data[intervention_date][location].keys():
                for model in data[intervention_date][location][outcome].keys():
                    return (intervention_date, location, outcome, model)
