import pickle
import os

# replace with imblearn's Pipeline...
from sklearn.pipeline import Pipeline

def read_pkl(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save2pkl(obj, filepath):
    with open(filepath, 'wb') as f:
        return pickle.dump(obj, f)

def make_subdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

rename_dict_estimator = {
    'logreg': 'Logistic Regression', 'nn': 'Neural Network',
    'svm': 'SVM', 'gb': 'Gradient Boosting', 'knn': 'K-Nearest Neighbors',
    'bayes': 'Naive Bayes', 'rf':'Random Forest', 'gp': 'Genetic Programming'
}; rename_dict_estimator_r = {value:key for key, value in rename_dict_estimator.items()}


def infer_estimator_name(estimator):
    # works for both imblearn.pipeline.Pipeline and sklearn.pipeline.Pipeline
    if issubclass(type(estimator), Pipeline):
        estimator = estimator.steps[-1][-1]
    return estimator.__repr__().split('(')[0]
