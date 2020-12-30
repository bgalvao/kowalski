import datetime
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV


from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif

from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import StackingClassifier


from scrpts.grid_search.pipelinestimators import *
from scrpts.grid_search.pipeline_modifiers import *

# meta params
UNSEEN_TEST_SIZE = 0.2
RANDOM_STATE = None

# -----------------------------------------------------------------------------
# set up test set

print('reloaded as')

def split_train_test(x, y, unseen_test_size=UNSEEN_TEST_SIZE, random_state=RANDOM_STATE):
    global x_train, y_train, x_test, y_test, train_idx, test_idx
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=unseen_test_size, random_state=random_state
    )
    for train_idx, test_idx in splitter.split(x, y):
        x_train, y_train = x.iloc[train_idx], y[train_idx]
        x_test, y_test = x.iloc[test_idx], y[test_idx]
        return x_train, y_train, x_test, y_test, train_idx, test_idx


# -----------------------------------------------------------------------------
def make_cvgen(n_splits=30, test_size=.8, random_state=RANDOM_STATE):
    return StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )  # can fix a random state


def make_cv_search_grid(estimator, param_grid, cross_generator):
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cross_generator,
        n_jobs=3,
        # scoring=["roc_auc", "accuracy", "f1", "precision", "recall"],
        scoring=["roc_auc", "accuracy", "f1", "precision", "recall"],
        refit="roc_auc",
        verbose=1,
    )



# a
# -----------------------------------------------------------------------------
def make_results_d():
    now = datetime.datetime.now()
    nau = now.__str__()

    results = {
        "meta": {
            "datetime": nau,
            "test_split": UNSEEN_TEST_SIZE,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "cv_gen": make_cvgen(),
            "random_state": RANDOM_STATE,
        }
    }

    pkl_save_name = "../results/{}.pkl".format(nau[:16]).replace(" ", "_")

    return results, pkl_save_name



# # -----------------------------------------------------------------------------
def stack_ensemble(estimators_list):
    return StackingClassifier(
        estimators_list,
        n_jobs=-1,
        verbose=9,
        # when final_estimator is None, it uses a LogisticRegression as final model
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )

if __name__ == '__main__':
    print('test')
