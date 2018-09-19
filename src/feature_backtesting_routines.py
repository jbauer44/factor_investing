import pandas as pd
import warnings


def add_bins_col_to_rank_df(df_feature,
                            n_bins,
                            bin_no_col='bin_no',
                            item_rank_col='equity_rank',
                            max_rank_col='max_rank'
                            ):
    """
    Description: This function takes as input a dataframe with ranks,
                 and creates a column with the respective bin number computed from the rank.

    :param df_feature:Type pandas dataframe. feature and period level dataframe with rank values.
    :param n_bins:Type int. number of bins to split the equities into.
    :param bin_no_col:Type str. bin number column name.
    :param item_rank_col:Type str. individual item rank column name.
    :param max_rank_col:Type str. maximum possible rank column name.
    :return:Type pandas dataframe. feature and period level dataframe with bin assignments.
    """

    df_feature[bin_no_col] = 1 + (n_bins * (df_feature[item_rank_col] - 1) // df_feature[max_rank_col])

    return df_feature


def get_ranks(df_feature,
              date_col_name,
              feature):
    """
    Description: This function takes as input a long dataframe with date and a feature value.
                 Ranks are computed for the set of feature values within in a date, and saved as a column.
                 The maximum possible rank is also added as a column. This is used to compute the bin number based on rank.

    :param df_feature:Type pandas dataframe. feature and period level dataframe with feature values.
    :param date_col_name:Type string. name of date column.
    :param feature:Type string. name of feature.
    :return:Type pandas dataframe. feature and period level dataframe with ranks assigned to the equities.
    """

    df_feature['equity_rank'] = df_feature.groupby(date_col_name)[feature].rank(method='min', ascending=False)
    df_feature['max_rank'] = df_feature.groupby(date_col_name)['equity_rank'].transform('max')

    return df_feature


def compute_date_level_metrics(df_detail,
                               bin_labels,
                               date_col_name,
                               return_col_name,
                               feature,
                               corr_method):
    """
    Description: This function computes the back-testing metrics, at the date level (freq of the input dataframe)


    :param df_detail:Type pandas dataframe. Detail dataframe with bins and rank information.
    :param bin_labels:Type list. list of bin labels. It is assumed that the labels are in descending order.
                      eg: ['Q1','Q2','Q3'] implies Q1 is the highest portfolio and Q3 is the lowest.
    :param date_col_name:Type str. Name of the date column.
    :param return_col_name:Type str. Name of the return column.
    :param feature:Type string. Name of feature.
    :param corr_method:Type string. correlation method being used.
    :return: Type pandas dataframe. feature and period level aggregate metrics.
    """

    df_agg = pd.DataFrame()

    for index, bin_lbl in enumerate(bin_labels):
        bin_no = index + 1
        bin_avg_lbl = bin_lbl + '_avg'
        bin_std_lbl = bin_lbl + '_std'

        df_agg[bin_avg_lbl] = df_detail[df_detail['bin_no'] == bin_no].groupby(date_col_name)[return_col_name].mean()
        df_agg[bin_std_lbl] = df_detail[df_detail['bin_no'] == bin_no].groupby(date_col_name)[return_col_name].std()

    highest_bin_avg_col = bin_labels[0] + '_avg'
    lowest_bin_idx = len(bin_labels) - 1
    lowest_bin_avg_col = bin_labels[lowest_bin_idx] + '_avg'

    df_agg['spread'] = (df_agg[highest_bin_avg_col] - df_agg[lowest_bin_avg_col])*100
    corr_object = df_detail.groupby(date_col_name)[[return_col_name, feature]].corr(method=corr_method)
    corr_series = corr_object.reset_index(1, drop=True)[return_col_name]
    corr_series = corr_series[~corr_series.eq(1)]
    df_agg['ic_cs'] = corr_series

    return df_agg


def get_dates_deviating_from_threshold(df_in,
                                       date_col_name,
                                       equity_identifier,
                                       items_per_bin_deviation_threshold,
                                       expected_no_of_items_per_bin):
    """
    Description: This function measures deviation in number of items per bin from the expected.
                 The months which deviate outside the acceptable threshold are flagged and output to the user.

    :param df_in:Type pandas dataframe. dataframe that is subset to rows relating to a certain bin number.
    :param date_col_name:Type str. Name of the date column.
    :param equity_identifier:Type str. Name of the equity identifier column.
    :param items_per_bin_deviation_threshold:Type int. Permissible deviation from the expected number of items per bin.
    :param expected_no_of_items_per_bin:Type int. Correct number of items per bin.
    :return:Type list. list of dates which deviate outside the acceptable threshold.
    """

    df_in_cnt = pd.DataFrame(df_in.groupby(date_col_name)[equity_identifier].count())

    df_in_cnt.rename(columns={equity_identifier: 'items_count'}, inplace=True)

    df_in_cnt['deviation'] = abs(df_in_cnt['items_count'] - expected_no_of_items_per_bin)

    threshold_deviation_mask = df_in_cnt['deviation'] > items_per_bin_deviation_threshold

    bad_dates = [item.date().strftime("%Y-%m-%d") for item in df_in_cnt[threshold_deviation_mask].index.tolist()]

    return bad_dates


def custom_formatwarning(message, category, filename, lineno, line=''):
    """
    Description: This is a custom function that overrides the default format warning function.
                 This will return only the warning message, without the code line details.

    :param message: Type str. Warning message.
    :return:        Type str. Returns only the warning message, without the code line details.
    """

    return str(message) + '\n'


def get_detail_backtest_results(input_df,
                                features,
                                return_col_name='returns',
                                equity_identifier='Equity Parent',
                                date_col_name='date',
                                n_bins=5,
                                bin_labels=None,
                                corr_method='spearman',
                                items_per_bin_deviation_threshold=1,
                                drop_months_outside_of_threshold=False):
    """
    Description: This function generates the back testing results for a list of features.

                 This procedure does not handle subsetting for a specified date range.
                 Subset to a specified date range needs to be done prior to passing the input dataframe.

                 The procedure works on the assumption that the bin_labels are specified in descending order.
                 eg: ['Q1','Q2','Q3','Q4'] implies Q1 is the highest portfolio and Q4 is the lowest.

                 The generation of bins work ideally with a sufficient number of unique values in a feature.
                 The items_per_bin_deviation_threshold parameter can be used to decide how strict we want to be with the effect of non-unique values.

                 items_per_bin_deviation_threshold acts on the difference between the expected number of items in a bin
                 vs the actual number of items.

                 drop_months_outside_of_threshold can be set to True, if the months deviating from the
                 above threshold should be excluded from back testing.

    :param input_df: Type pandas dataframe. long format dataframe.
    :param features: Type list. list of features for which backtesting needs to be perforrmed. These should correspond
                     to the names of the columns in the df_long dataframe.
    :param return_col_name: Type str. Name of the return column.
    :param equity_identifier : Type str. Name of the equity identifier column.
    :param date_col_name:Type str. Name of the date column.
    :param n_bins:Type int. number of bins to split the equities into.
    :param bin_labels:Type list. list of bin labels. It is assumed that the labels are in descending order.
                        eg: ['Q1','Q2','Q3'] implies Q1 is the highest portfolio and Q3 is the lowest.
    :param corr_method:Type string. correlation method being used.
    :param items_per_bin_deviation_threshold:Type int. Permissible deviation from the expected number of items per bin.
    :param drop_months_outside_of_threshold:Type boolean. Decision to drop months that break deviate beyond the acceptable
                                                          items_per_bin_deviation_threshold.

    :return:Type pandas dataframe. detail backtesting results for each period
    """

    if bin_labels is None:
        bin_labels = ['Q' + str(i + 1) for i in range(n_bins)]

    df_long = input_df.copy()
    long_cols = list(df_long.columns)

    if date_col_name not in long_cols:
        df_long = df_long.reset_index()
        df_long.rename(columns={'index': 'date'}, inplace=True)

    if return_col_name in features:
        features.remove(return_col_name)

    detail_results = []
    features = sorted(features)
    feature_cnt = 0
    total_features = len(features)
    print('Total features for processing: ' + str(total_features))

    warnings.formatwarning = custom_formatwarning

    for feature in features:

        category = feature.split('_bshift')[0]
        feature_cols = [equity_identifier, date_col_name, return_col_name, feature]

        df_feature_detail = df_long[feature_cols].copy()

        df_feature_detail = get_ranks(df_feature_detail,
                                      date_col_name,
                                      feature)

        df_feature_detail = add_bins_col_to_rank_df(df_feature_detail,
                                                    n_bins)

        df_bin_check = pd.DataFrame(df_feature_detail.groupby(date_col_name)['bin_no'].max())
        bin_check_mask = df_bin_check['bin_no'] != n_bins
        insufficient_bins_dates = [item.date().strftime("%Y-%m-%d") for item in df_bin_check[bin_check_mask].index.tolist()]

        if len(insufficient_bins_dates) > 0:

            warnings.warn('\nInsufficient bins warning:\nFeature: ' + feature+'\n'+'\n' +
                             'Months with insufficient bins:' + str(insufficient_bins_dates)+ '\n' + '\n' +
                             'These months are excluded from the back testing computation')

            df_feature_detail = df_feature_detail[~df_feature_detail[date_col_name].isin(insufficient_bins_dates)]
            print(df_feature_detail.shape)

        total_no_of_items = df_feature_detail[equity_identifier].unique().shape[0]
        expected_no_of_items_per_bin = total_no_of_items/n_bins

        mask_bin_lowest = df_feature_detail['bin_no'] == 1
        mask_bin_highest = df_feature_detail['bin_no'] == n_bins

        df_bin_lowest = df_feature_detail[mask_bin_lowest].copy()
        df_bin_highest = df_feature_detail[mask_bin_highest].copy()

        bin_lowest_bad_dates = get_dates_deviating_from_threshold(df_bin_lowest,
                                                                  date_col_name,
                                                                  equity_identifier,
                                                                  items_per_bin_deviation_threshold,
                                                                  expected_no_of_items_per_bin)

        bin_highest_bad_dates = get_dates_deviating_from_threshold(df_bin_highest,
                                                                   date_col_name,
                                                                   equity_identifier,
                                                                   items_per_bin_deviation_threshold,
                                                                   expected_no_of_items_per_bin)

        if len(bin_lowest_bad_dates) > 0 or len(bin_highest_bad_dates) > 0:

            warnings.warn('\nDeviation from threshold warning:\nFeature: ' + feature+'\n'+'\n' +
                            'Top Portfolio - Months which deviate from threshold: '+str(bin_highest_bad_dates)+'\n'+'\n' +
                            'Bottom Portfolio - Months which deviate from threshold: '+str(bin_lowest_bad_dates))

            if drop_months_outside_of_threshold:
                months_to_drop = bin_lowest_bad_dates + bin_highest_bad_dates

                warnings.warn('\nMonths dropped warning:\nFeature: ' + feature + '\n'+'\n' +
                                'Months: '+str(months_to_drop) +' will be dropped from computation')

                df_feature_detail = df_feature_detail[~df_feature_detail[date_col_name].isin(months_to_drop)]

        df_feature_detail_agg = compute_date_level_metrics(df_feature_detail,
                                                           bin_labels,
                                                           date_col_name,
                                                           return_col_name,
                                                           feature,
                                                           corr_method)

        df_feature_detail_agg['feature'] = feature

        df_feature_detail_agg['category'] = category

        detail_results.append(df_feature_detail_agg)

        feature_cnt += 1

        if feature_cnt % 100 == 0:
            print(str(feature_cnt) + ' features completed')

    detail_results_df = pd.concat(detail_results)

    return detail_results_df


def perform_aggregation_across_time(detail_results):
    """
    Description: This function takes as input the monthly level back testing results, and
                 returns a data frame with the results aggregated across time.

    :param detail_results: Type pandas dataframe. detail results that need to be aggregated.
    :return:Type pandas dataframe. Aggregated dataframe.
    """
    out_df = pd.DataFrame()

    all_cols = list(detail_results.columns)
    cols_for_std = ['spread', 'ic_cs', 'Qe', 'mc_return']
    key_cols = ['feature', 'category']
    cols_for_avg = sorted(list(set(all_cols) - set(cols_for_std) - set(key_cols)))

    for col in cols_for_avg:
        out_df[col] = detail_results.groupby(key_cols)[col].mean()

    for col in cols_for_std:
        if 'avg' not in col:
            avg_col = col + '_avg'
            std_col = col + '_std'
        else:
            avg_col = col
            std_col = col.split('_')[0] + '_std'

        out_df[avg_col] = detail_results.groupby(key_cols)[col].mean()
        out_df[std_col] = detail_results.groupby(key_cols)[col].std()

    return out_df
