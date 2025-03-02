from typing import Dict, List

import pandas as pd

import pandas as pd


def to_pickled_df(data: List, index_column_name: str, pkl_file_name: str) -> pd.DataFrame:
    # Ensure all items in the data list are DataFrames or Series
    valid_data = [d for d in data if isinstance(d, (pd.DataFrame, pd.Series))]

    # If no valid DataFrame/Series objects found, raise an error
    if not valid_data:
        raise TypeError("All objects in 'data' must be either pandas DataFrames or Series.")

    # Concatenate the valid data
    combined_df = pd.concat(valid_data, axis=0)
    combined_df.set_index(index_column_name, inplace=True)
    combined_df.reset_index(drop=False, inplace=True)

    # Save to pickle file
    combined_df.to_pickle(pkl_file_name + ".pkl")
    return combined_df


def to_dataframe(column_name: str, column_value: str, results: Dict[str, float]) -> pd.DataFrame:
    """Convert results into a pandas DataFrame."""
    results[column_name] = column_value
    df = pd.DataFrame([results])
    df = df[[column_name] + [col for col in df.columns if col != column_name]]
    return df
