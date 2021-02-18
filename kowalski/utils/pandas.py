import pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon

def highlight_significant_pval(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_significant = (s < .05)
    return ['background-color: yellow' if v else '' for v in is_significant]


# TODO: add multiple comparisons correction of some sort.
def gb_apply_wilcox(group_df, cat_col, metric_col, significance_lvl=.05):
    """Applies a wilcoxon test comparing values of <metric_col> of the
    different <cat_col> categories present in the group, in the fashion of
    a triangular matrix.

    This function is meant to be passed to pandas.DataFrameGroupBy.apply.

    For example:

    ```python
    df.groupby('dataset').apply(gb_apply_wilcox, cat_col='fs_method', metric_col='f1')
    ```


    Args:
        group_df (DataFrame): DataFrame containing the group.
        cat_col (str): Column in DataFrame that specifies the dimension to compare.
        metric_col (str): The numeric values to be tested.

    Returns:
        DataFrame: DataFrame in triangular matrix function.
    """
    group_result_df = pd.DataFrame()
    for ha, hb in combinations(group_df[cat_col].unique(), 2):
        stat, pval = wilcoxon(
            group_df[group_df[cat_col] == ha][metric_col],
            group_df[group_df[cat_col] == hb][metric_col]
        )
        group_result_df.loc[ha, hb] = pval

    return group_result_df
