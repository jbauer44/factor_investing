"""
Functions useful in finance related applications

"""

import numpy as np
import pandas as pd

import datetime
import dateutil.relativedelta as relativedelta


def project_to_first(dt):
    return datetime.datetime(dt.year, dt.month, 1)


def multiple_returns_from_levels_vec(df_in, period=1):

    df_out = df = (df_in - df_in.shift(period)) / df_in.shift(period)

    return df_out


def df_restrict_dates(df_in, start_date, end_date, multi_index=False, date_name='date'):

    """
    restrict input dataframe to certain date range
    boundaries are inclusive
    index must be in date format

    :param df_in: pandas data frame, index must be in datetime format; can deal with multi-index now as well
    :param start_date: datetime.datetime (date or certain string formats might also work)
    :param end_date: datetime.datetime (date or certain string formats might also work)
    :return: reduced dateframe
    """

    df_out = df_in.copy()
    if multi_index:
        mask = (df_out.index.get_level_values(date_name) >= start_date) & \
               (df_out.index.get_level_values(date_name) <= end_date)
    else:
        mask = (df_out.index >= start_date) & (df_out.index <= end_date)
    return df_out[mask]


def levels_from_returns(df_in, infield='return', outfield='level', starting_level=1, frequency='daily',
                        initial_date=None):
    assert frequency in ['daily', 'monthly', 'quarterly'], 'not implemented'
    start_date = df_in.index.min()
    df_out = df_in[[infield]].copy()

    if initial_date is None:
        if frequency == 'daily':
            initial_date = start_date + relativedelta.relativedelta(days=-1)
        if frequency == 'monthly':
            initial_date = start_date + relativedelta.relativedelta(months=-1)
        if frequency == 'quarterly':
            initial_date = start_date + relativedelta.relativedelta(months=-3)

    df_out.loc[initial_date] = starting_level
    df_out.sort_index(ascending=True, inplace=True)
    df_out[outfield + '_temp'] = compute_levels(starting_level, df_in[infield].values)
    df_out.drop(infield, axis=1, inplace=True)
    df_out.rename(columns={outfield + '_temp': outfield}, inplace=True)
    return df_out

def monthly_returns(df_in, field='Close', out_name='monthly_return', day_of_month='last'):
    assert day_of_month in ['first', 'last'], 'not implemented'
    start_date = df_in.index.min()
    end_date = df_in.index.max()

    shift_start_date = start_date + relativedelta.relativedelta(months=1)
    first_date_returns = datetime.datetime(shift_start_date.year, shift_start_date.month, 1)
    last_date_returns = datetime.datetime(end_date.year, end_date.month, 1)

    date = first_date_returns
    l_monthly_returns = []
    l_dates = []
    while date <= last_date_returns:
        this_year = date.year
        this_month = date.month
        final_day = find_day_in_month(df_in.index, this_year, this_month, which=day_of_month)
        mask = df_in.index == final_day
        final_val = df_in[mask][field].iloc[0]
        prev_date = date + relativedelta.relativedelta(months=-1)
        prev_year = prev_date.year
        prev_month = prev_date.month
        initial_day = find_day_in_month(df_in.index, prev_year, prev_month, which=day_of_month)
        mask = df_in.index == initial_day
        prev_val = df_in[mask][field].iloc[0]
        #print(prev_initial_day, prev_val)
        if abs(prev_val) > 0.0:
            monthly_return = (final_val - prev_val) / prev_val
        else:
            monthly_return = np.nan
        l_monthly_returns.append(monthly_return)
        l_dates.append(date)
        date += relativedelta.relativedelta(months=1)

    df_out = pd.DataFrame({out_name: l_monthly_returns}, index=l_dates)
    df_out.index.name = df_in.index.name
    return df_out

def compute_levels(l0, returns):
    levels = [l0]
    for k in range(len(returns)):
        levels.append(levels[k]*(1.0 + returns[k]))
    return np.array(levels)
