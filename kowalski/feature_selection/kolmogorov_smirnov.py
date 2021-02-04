
from scipy.stats import ks_2samp
import pandas as pd

# this cannot be used... because "my" signature demands train_x and test_x...
# from sklearn.base import TransformerMixin, BaseEstimator

class KolmogorovSmirnovFilter:


    def __init__(self, min_ks_pval:float=.1):
        self.min_ks_pval = min_ks_pval


    def fit(self, train_x, test_x):

        assert (train_x.columns == test_x.columns).all()
        
        results = {'ks_statistic':[], 'pvalue':[]}
        for feature in train_x.columns:
            result = ks_2samp(train_x[feature], test_x[feature])
            results['ks_statistic'].append(result.statistic)
            results['pvalue'].append(result.pvalue)

        self.results_ = pd.DataFrame(
            results,
            index=train_x.columns
        ).sort_values('pvalue', ascending=False)

        self.stable_features_ = \
            list(self.results_[self.results_.pvalue >= self.min_ks_pval].index)
        
        return self


    def transform(self, train_x, test_x):
        return train_x[self.stable_features_], test_x[self.stable_features_]


    def fit_transform(self, train_x, test_x):
        self.fit(train_x, test_x)
        return self.transform(train_x, test_x)
