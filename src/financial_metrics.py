"""
functions to evaluate portfolio/index metrics


"""

#general

import dateutil.relativedelta as relativedelta

import pandas as pd
import numpy as np

from .finance_functions import levels_from_returns, monthly_returns


def extract_performance(df_in):
    #df_in =   df_index_comb
    first_date = df_in.index.min()
    last_date = df_in.index.max()
    tdel = relativedelta.relativedelta(last_date, first_date)
    n_years = tdel.years + tdel.months/12
    total_return = (df_in.loc[last_date] - df_in.loc[first_date])/df_in.loc[first_date]
    df_result = pd.DataFrame(total_return, columns=['Total Return']).T
    df_result.loc['Ave Annual Return'] = ((df_in.loc[last_date]/df_in.loc[first_date])**(1/n_years) - 1.0).T
    return df_result


def extract_performance_from_returns(df_returns, col_name='returns', out_name='index'):
    exp_returns = df_returns[col_name].mean()
    df_result = pd.DataFrame(np.nan, columns=[out_name], index=['mean_returns'])
    df_result.loc['mean_returns'] = exp_returns
    vol = df_returns[col_name].std()
    df_result.loc['volatility'] = vol
    df_result.loc['Sharpe_ratio'] = exp_returns / vol
    cum_return = levels_from_returns(df_returns, col_name, frequency='monthly')['level'][-1] - 1.0
    df_result.loc['cumulative_returns'] = cum_return

    first_date = df_returns.index.min()
    last_date = df_returns.index.max()
    tdel = relativedelta.relativedelta(last_date, first_date)
    n_years = tdel.years + tdel.months / 12
    df_result.loc['ave_annual_return'] = (1.0 + cum_return) ** (1 / n_years) - 1.0

    return df_result


def multiple_extract_performance(d_return, start_date=None, end_date=None):
    for k, key in enumerate(d_return.keys()):
        df_in = d_return[key]['df']
        col_name = d_return[key]['col_name']
        out_name = d_return[key]['out_name']
        if start_date is not None and end_date is not None:
            df_in = df_restrict_dates(df_in, start_date, end_date)
        df_temp = extract_performance_from_returns(df_in, col_name=col_name, out_name=out_name)
        if k == 0:
            df_result = df_temp
        else:
            df_result = df_result.merge(df_temp, left_index=True, right_index=True)
    return df_result
