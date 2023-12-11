from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_string_dtype, is_categorical_dtype
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sksurv.datasets import load_flchain


def get_full_df_from_xy(X: pd.DataFrame, y: np.ndarray):
    res_df = X.copy()
    res_df['event'] = [pair[0] for pair in y]
    res_df['duration'] = [pair[1] for pair in y]
    return res_df


def get_flchain() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(*load_flchain(), train_size=0.7, random_state=42)

    df_train = get_full_df_from_xy(X=x_train, y=y_train)
    df_test = get_full_df_from_xy(X=x_test, y=y_test)

    le_dict: Dict[str, LabelEncoder] = {
        key: LabelEncoder() for key in x_train.keys()
        if is_string_dtype(x_train[key]) or is_categorical_dtype(x_train[key])
    }

    for key, le in le_dict.items():
        x_train[key] = le.fit_transform(x_train[key])
        x_test[key] = le.fit_transform(x_test[key])

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train = imp.fit_transform(x_train)
    x_test = imp.transform(x_test)

    return x_train, y_train, x_test, y_test


def get_fl_chain_full() -> pd.DataFrame:
    df = get_full_df_from_xy(*load_flchain())
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    df['creatinine'] = imp.fit_transform(df[['creatinine']])
    return df
