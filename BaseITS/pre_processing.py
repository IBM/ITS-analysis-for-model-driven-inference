import pandas as pd
import numpy as np
from datetime import datetime


def align_prophet_naming_convection(
    df: pd.DataFrame, date_col_name: str, y_col_name: str, verbose=False
):
    """Function to align column names with ones expected by prophet model

    Args:
        df (pd.DataFrame): dataframe with the columns
        x_col_name (str): outcome column
        y_col_name (str): date column

    Returns:
        pd.DataFrame: dataframe with renamed columns to the expected prophet naming convection
    """
    if date_col_name not in df.columns.tolist():
        return ValueError(
            "Make sure that the provided date column name are in the dataframe provided."
        )
    if y_col_name not in df.columns.tolist():
        return ValueError(
            "Make sure that the provided y column name are in the dataframe provided."
        )
    elif date_col_name == "ds" and y_col_name == "y":
        if verbose:
            print("Column names already in the required convection")
        return df

    elif date_col_name != "ds" and y_col_name == "y":
        if verbose:
            print(
                "Only y column name in the required convection. Converting date column to required convection"
            )
        df.rename(columns={date_col_name: "ds"}, inplace=True)
        return df

    elif date_col_name == "ds" and y_col_name != "y":
        if verbose:
            print(
                "Only date column name in the required convection. Converting y column to required convection"
            )
        df = df.rename(columns={y_col_name: "y"})
        return df

    elif date_col_name != "ds" and y_col_name != "y":
        if verbose:
            print(
                "None of the column names in the required convection. Converting y column to required convection"
            )
        df.rename(columns={date_col_name: "ds"}, inplace=True)
        df.rename(columns={y_col_name: "y"}, inplace=True)
        return df

    ## Add test if column names provided not in the dataframes themselves

    else:
        raise ValueError(
            "Not covered test case. Should be investigated for further code improvement"
        )


def str_date_validate(date_text: str):
    """Function to validate strings that they are in the correct datetime format for conversion.

    Args:
        date_text (str): String with the date

    Raises:
        ValueError: Raises an error incase wrong string date format is provided

    Returns:
        datetime: Datetime converted value in the format ( '%Y-%m-%d')

    """
    try:
        value = datetime.datetime.strptime(date_text, "%Y-%m-%d")
        return value
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def dates_validation(df: pd.DataFrame, date_col_name: str):
    """Function to validate dates to datetime format

    Args:
        df (pd.DataFrame): Dataframe with the data
        date_col_name (str): column with the dates

    Returns:
        pd.series: Series with the date_col_name with datetime datatype
    """

    # try: check if format is the expected one.

    if df[date_col_name].dtype != "datetime64[ns]":

        df[date_col_name] = pd.to_datetime(df[date_col_name])

        if str(df[date_col_name][0]).startswith("1970"):
            return "Error converting, make sure the datetime column is in the YYYY-MM-DD format first"
        else:
            return df

    elif df[date_col_name].dtype == "datetime64[ns]":
        df[date_col_name] = pd.to_datetime(df[date_col_name], format="%Y-%m-%d")
        print("Date column already in the correct format")
        return df

    else:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def aggregation_wide_df_type(
    df: pd.DataFrame,
    location_col_name: str,
    date_col_name: str,
    outcome_cols: list,
):
    """Function to aggregate outcome values in a wide dataframe type based on the date, outcome and location

    Args:
        df (pd.DataFrame): Wide dataframe type
        location_col_name (str): column name of the location in the dataframe
        date_col_name (str): date column name in the dataframe
        outcome_cols (list): list of the outcome column names

    Returns:
        pd.DataFrame: Dataframe with aggregated counts per location, date and outcome
    """

    df = (
        df.groupby([date_col_name, location_col_name])[outcome_cols].sum().reset_index()
    )
    # regions_df.groupby(['periodname','Region'])['Antenatal 4th Visit'].sum().reset_index()

    return df


def aggregation_long_df_type(
    df: pd.DataFrame,
    location_col_name: str,
    date_col_name: str,
    outcome_col_name: str,
    outcome_value_col_name: str,
):
    """Function to aggregate outcome values in a long dataframe type based on the date, outcome and location

    Args:
        df (pd.DataFrame): Long dataframe with the data
        location_col_name (str): column name of the locations in the dataframe
        date_col_name (str): column name of the date in the dataframe
        outcome_col_name (str): column name of the outcome in the dataframe
        outcome_value_col_name (str): column name of the outcome values in the dataframe

    Returns:
        pd.DataFrmae: Dataframe with aggregated counts per location, date and outcome
    """

    df = (
        df.groupby([location_col_name, date_col_name, outcome_col_name])[
            outcome_value_col_name
        ]
        .sum()
        .reset_index()
    )
    # long_df.groupby(['region','outcome','ds'])['value'].sum().reset_index()

    return df


def create_log_offset(df: pd.DataFrame, ofset_column: str):
    """Create offset for the poisson regression forecast model: (log)

    Args:
        df (pd.DataFrame): dataframe with the offset column
        ofset_column (str): column of the offset

    Returns:
        pd.Series: calcutated offset
    """
    offset = np.log(df[ofset_column])

    return offset
