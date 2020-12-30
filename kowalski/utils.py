import pickle


from scipy.stats import bayes_mvs, wilcoxon, t, sem
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, \
    precision_score, recall_score, cohen_kappa_score, roc_curve, make_scorer,\
    matthews_corrcoef

import numpy as np

def read_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save2pkl(obj, filepath):
    with open(filepath, 'wb') as f:
        return pickle.dump(obj, f)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def lcb(series):
    return bayes_mvs(series, .95)[0].minmax[0]

def ucb(series):
    return bayes_mvs(series, .95)[0].minmax[1]

def wilcoxon_pvalue(series):
    return wilcoxon(series-series.mean()).pvalue

def specificity(y_true, y_pred):
    # https://uberpython.wordpress.com/2012/01/01/precision-recall-sensitivity-and-specificity/
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp)

def w_recall_spec(y_true, y_pred):
    return .4*recall_score(y_true, y_pred) + .6*specificity(y_true, y_pred)
    

kappa_score = make_scorer(cohen_kappa_score)
mcc_score = make_scorer(matthews_corrcoef)
specificity_score = make_scorer(specificity)

scorers = {
    'roc_auc': 'roc_auc',
    'kappa': kappa_score,
    'accuracy': 'balanced_accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'specificity': specificity_score,
    'mcc': mcc_score
}


rename_dict_metric = {
    'roc_auc': 'ROC AUC',
    'acc': 'Accuracy',
    'f1': 'F1',
    'prec': 'Precision',
    'acc': 'Accuracy',
    'rec': 'Recall/Sensitivity',
    'spec': 'Specificity',
    'kappa': 'Cohen\'s Kappa',
    'mcc': 'Matthew\'s Correlation Coefficient'
}

rename_dict_statistic = {
    'lcb': 'LCI',
    'ucb': 'UCI',
    'mean': 'Mean',
    'wilcoxon_pvalue': 'Wilcoxon p-value'
}

rename_dict_estimator = {
    'logreg': 'Logistic Regression', 'nn': 'Neural Network',
    'svm': 'SVM', 'gb': 'Gradient Boosting', 'knn': 'K-Nearest Neighbors',
    'bayes': 'Naive Bayes', 'rf':'Random Forest'
}; rename_dict_estimator_r = {value:key for key, value in rename_dict_estimator.items()}


metric_cols = ['roc_auc', 'acc', 'kappa', 'f1', 'prec', 'rec', 'spec']


