from sklearn.feature_selection import VarianceThreshold, SelectFdr
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np


"""
TODO

- variance thresh
- correlation thresh  # and I kind of do not like the way features are eliminated
- multiple comparisons correction

"""


class CorrelationThreshold(BaseEstimator, TransformerMixin):


    def __init__(self, threshold=.9, untie=None, feature_names=None):
        
        self.threshold = threshold
        self.untie = untie
        self.feature_names = feature_names


    """
    File "/home/bee/Projects/binder/repo/lab/notebooks/scrpts/feature_selection/prefilters.py", line 36, in fit
        upper = np.triu(corr_matrix, k=1)  # sklearn checker bugs out here...
    File "<__array_function__ internals>", line 6, in triu
    File "/home/bee/.anaconda3/envs/causal/lib/python3.7/site-packages/numpy/lib/twodim_base.py", line 463, in triu
        mask = tri(*m.shape[-2:], k=k-1, dtype=bool)
    TypeError: tri() missing 1 required positional argument: 'N'
    """

    def fit(self, X, y=None):

        # https://scikit-learn.org/stable/developers/develop.html#universal-attributes
        self.n_features_in_ = X.shape[1]

        corr_matrix = np.abs(np.corrcoef(X, rowvar=False))
        upper = np.triu(corr_matrix, k=1)  # sklearn checker bugs out here...

        if self.untie == None:
            # Find index of feature columns with correlation greater than set threshold
            self.support_ = ~np.any((upper >= self.threshold), axis=0)
        else:
            raise NotImplementedError
        return self


    def fit_transform(self, X, y=None):
        self.fit(X)
        return X[:, self.support_]

    def transform(self, X):
        return X[:, self.support_]

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {'threshold': self.threshold, 'untie': self.untie, 'feature_names': self.feature_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self



class CorrelationThresholdPandas(BaseEstimator, TransformerMixin):


    def __init__(self, threshold=.9, untie=None, feature_names=None):
        
        self.threshold = threshold
        self.untie = untie
        self.feature_names = feature_names


    def fit(self, X, y=None):

        # Create correlation matrix
        corr_matrix = X.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        if self.untie == None:
            # Find index of feature columns with correlation greater than set threshold
            self.to_drop_ = [column for column in upper.columns if any(upper[column] > self.threshold)]
        else:
            raise NotImplementedError
        return self


    def fit_transform(self, X, y=None):
        self.fit(X)
        return X.drop(self.to_drop_, axis=1)

    def transform(self, X):
        return X.drop(self.to_drop_, axis=1)

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {'threshold': self.threshold, 'untie': self.untie, 'feature_names': self.feature_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self




if __name__ == '__main__':

    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(CorrelationThreshold())

    