from typing import Dict, List

import pandas as pd


def to_pickled_df(indicator_data: List, pkl_file_name: str) -> pd.DataFrame:
    combined_df = pd.concat(indicator_data, axis=0)
    combined_df.set_index('Ticker', inplace=True)
    combined_df.reset_index(drop=False, inplace=True)
    combined_df.to_pickle(pkl_file_name + ".pkl")
    return combined_df
