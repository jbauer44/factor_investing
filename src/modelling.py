"""

Helper functions for modelling

"""

import numpy as np
import pandas as pd


# statistics and machine learning

from sklearn.preprocessing import Imputer


# settings
np.random.seed(1234)


# fill missing values

def df_fill_missing(df_in, missing_val_strategy='drop'):
    miss_strategies = ['ffill', 'bfill', 'drop', 'mean', 'median']
    assert missing_val_strategy in miss_strategies, 'not implemented'

    df_out = df_in.copy()
    if missing_val_strategy == 'bfill':
        df_out.bfill(inplace=True)
    if missing_val_strategy == 'ffill':
        df_out.ffill(inplace=True)

    if missing_val_strategy == 'drop':
        df_out.dropna(inplace=True)

    if missing_val_strategy in ['mean', 'median']:
        imputer = Imputer(strategy=missing_val_strategy)
        df_out = pd.DataFrame(imputer.fit_transform(df_out), columns=df_out.columns, index=df_out.index)

    return df_out

