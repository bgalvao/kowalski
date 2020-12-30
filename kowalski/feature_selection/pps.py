
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit,\
    cross_validate

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
    mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import pandas as pd
import numpy as np

from scipy.stats import mode


_FLOAT_FEATURE = 'FLOAT'
_CATEGORICAL_FEATURE = 'CATEGORICAL'
_ORDINAL_FEATURE = 'ORDINAL'


_REGRESSION_PROBLEM = 'REGRESSION'
_CLASSIFICATION_PROBLEM = 'CLASSIFICATION'

classification_metrics = ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']
regression_metrics = ['neg_mean_absolute_error', 'neg_mean_squared_error']

str_metrics = {
    'roc_auc': roc_auc_score,
    'neg_mean_squared_error': mean_squared_error,
    'neg_mean_absolute_error': mean_absolute_error
}


def _determine_feature_type(feature_series, standardize_if_numerical=True):
    """
    If feature_series is of type string (even strings representing integers),
    then it is a categorical feature and it will be one-hot encoded.

    Otherwise, it will just be standardized, being a numerical feature.
    """
    if feature_series.dtype == np.dtype('O'):  # string
        return OneHotEncoder().fit_transform(feature_series.values.reshape(-1, 1))

    elif feature_series.dtype == float and standardize_if_numerical:
        return StandardScaler().fit_transform(feature_series.values.reshape(-1, 1))
    
    else:
        return feature_series.values.reshape(-1, 1)



def _determine_target_type(target_series):
    if target_series.dtype == np.dtype('O'):
        return LabelEncoder().fit_transform(target_series).ravel(), _CLASSIFICATION_PROBLEM
    else:
        return target_series, _REGRESSION_PROBLEM


def ppscore(df, y:str=None, metric='roc_auc', cvgen=StratifiedShuffleSplit(test_size=.2, n_splits=30)):

    y_target, problem_type = _determine_target_type(df[y])

    if problem_type == _REGRESSION_PROBLEM:
        raise NotImplementedError

    estimator = DecisionTreeRegressor() if problem_type == _REGRESSION_PROBLEM \
        else DecisionTreeClassifier()

    # get estimator scores on test set.
    estimator_scores = {}
    for col in df.columns:
        if col != y:
            x = _determine_feature_type(df[col])
            estimator_scores[col] = cross_validate(
                estimator, x, y_target, scoring=metric, cv=cvgen, n_jobs=2
            )
    estimator_scores = {col: d['test_score'] for col, d in estimator_scores.items()}
    estimator_scores = pd.DataFrame(
        data=estimator_scores,
        index=np.arange(cvgen.n_splits)
    ).stack().swaplevel()
    estimator_scores.index.rename(['x', 'split'], inplace=True)
    estimator_scores = estimator_scores.reset_index().rename(columns={0:'estimator_score'})
    # return estimator_score
    estimator_scores = estimator_scores.set_index('split')
    # return estimator_scores
    

    # get baseline scores on test set
    baseline_scores = []
    for train_idx, test_idx in cvgen.split(x, y_target):

        naive_prediction = np.median(y_target[train_idx]) \
            if problem_type == _REGRESSION_PROBLEM else mode(y[train_idx])
        naive_prediction = pd.Series(naive_prediction, index=test_idx)  # setting it to the right length

    
        if problem_type == _REGRESSION_PROBLEM:
            naive_prediction = np.median(y[train_idx])
            # setting it to the right length
            naive_prediction = pd.Series(naive_prediction, index=test_idx)
            naive_prediction_score = mean_absolute_error(y[test_idx], naive_prediction)
            baseline_scores.append(naive_prediction)

        else:
            naive_prediction = mode(y[train_idx])
            n_classes = pd.Series(y).nunique()

            naive_proba = np.zeros(shape=(len(test_idx), n_classes))
            naive_proba[:, naive_prediction] = 1.0

            naive_prediction_score = roc_auc_score(
                y_true=y[test_idx], y_score=naive_proba
            )
            baseline_scores.append(naive_prediction_score)

            # if you want to make
            # naive_prediction = pd.Series(naive_prediction, index=test_idx)
            # naive_prediction_score = accuracy_score(y[test_idx], naive_prediction)

    dfr = pd.DataFrame(data={'baseline_score':baseline_scores}, index=np.arange(cvgen.n_splits))
    dfr.index.name = 'split'
    dfr['y'] = y
    
    dfpps = dfr.join(estimator_scores).reset_index().set_index(['x', 'y', 'split'])

    if problem_type == _REGRESSION_PROBLEM:
        raise NotImplementedError
    else:
        dfpps['pps'] = (dfpps['estimator_score'] - dfpps['baseline_score']) /\
            (1.0 - dfpps['baseline_score'])

    dfpps['estimator'] = estimator.__str__()
    return dfpps

    
        
        

if __name__ == '__main__':

    test = pd.read_csv('dfx.csv')
    ppscore(test, 'response')



        



