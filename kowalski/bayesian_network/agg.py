from scipy.stats import bayes_mvs, wilcoxon


def lower_bound(series):
    return bayes_mvs(series, alpha=.95)[0].minmax[0]
    
def upper_bound(series):
    return bayes_mvs(series, alpha=.95)[0].minmax[1]

def wilcoxon_pvalue(series):
    return wilcoxon(series-series.mean()).pvalue


renamer_stats = {
    'lower_bound': 'Lower Confidence Bound',
    'mean': 'Mean',
    'upper_bound': 'Upper Confidence Bound',
    'std': 'Standard Deviation',
    'wilcoxon p_value': 'Wilcoxon p-value on Mean'
}

def aggregate_results(data):
    # aggregates each column in data..
    return data.agg([
        lower_bound, 'mean', upper_bound, 'std', wilcoxon_pvalue
    ]).rename(index=renamer_stats)