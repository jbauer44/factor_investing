"""
Functionality for index weight adjustment etc.



"""

# libraries

#general

import dateutil.relativedelta as relativedelta

import numpy as np
import pandas as pd

from .finance_functions import levels_from_returns


def index_levels_from_returns(df_weights, df_returns, starting_level=100, out_field='index_name',
                              transaction_costs=True, cost_percentage=0.005, frequency='monthly'):
    assert frequency in ['daily', 'monthly', 'quarterly'], 'not implemented'

    df_index_returns = pd.DataFrame((df_returns * df_weights).sum(axis=1), columns=['return'])
    if transaction_costs:

        # weights after period
        df_weights_actual = compute_actual_weights(df_weights, df_returns)
        # difference, if negative we have to buy asset
        df_weights_diff = weight_diff_actual_target(df_weights_actual, df_weights)

        start_date = df_returns.index.min()

        if frequency == 'daily':
            initial_date = start_date + relativedelta.relativedelta(days=-1)
        if frequency == 'monthly':
            initial_date = start_date + relativedelta.relativedelta(months=-1)
        if frequency == 'quarterly':
            initial_date = start_date + relativedelta.relativedelta(months=-3)

        new_index = [initial_date] + list(df_returns.index)
        df_index_levels = pd.DataFrame(np.nan, columns=[out_field], index=new_index)

        value = starting_level
        df_index_levels.loc[initial_date, out_field] = starting_level
        for date in df_returns.index:
            # value before rebalancing
            # value = (1 + df_index_returns.loc[date].get_value('return')) * value
            value = (1 + df_index_returns.at[date, 'return']) * value
            proportion_buy = df_weights_diff.loc[date][df_weights_diff.loc[date] < 0.0].sum()
            # print(proportion_buy)
            buy_amount = abs(value * proportion_buy)
            # value after rebalancing; only consider bid-ask spread costs for buying
            value = value - buy_amount * cost_percentage
            df_index_levels.loc[date, out_field] = value

    else:
        df_index_levels = levels_from_returns(df_index_returns, infield='return',
                                              outfield=out_field, starting_level=starting_level,
                                              frequency=frequency)
    return df_index_levels


