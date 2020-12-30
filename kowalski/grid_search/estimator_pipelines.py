from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest

from imblearn.pipeline import Pipeline
import numpy as np

RANDOM_STATE = None

# -------------------------- bayes
pipeline_bayes = Pipeline([("estimator", GaussianNB())])
param_grid_bayes = {"estimator__var_smoothing": [0.05, 0.1, 0.15, 0.2, 0.25]}


# -------------------------- knn
pipeline_knn = Pipeline([("estimator", KNeighborsClassifier(metric="euclidean"))])
param_grid_knn = {
    "estimator__n_neighbors": [4, 5, 6, 7, 8, 9, 10],
    "estimator__weights": ["uniform", "distance"],
    "estimator__p": [1, 2, 3],  # parameter p of minkowski distance
}


# -------------------------- log reg
pipeline_logreg = Pipeline(
    [("estimator", LogisticRegression(
        max_iter=50000, random_state=RANDOM_STATE, solver='saga'
    ))]
)

param_grid_logreg = [
    {
        "estimator__penalty": ["elasticnet"],
        "estimator__l1_ratio": np.arange(0.0, 1.1, 0.1),  # i.e. spectrum from l2 to l1
        "estimator__fit_intercept": [True, False],
        # "estimator__class_weight": [(.55, .45), (.6, .4), (.65, .35), (.7, .3)]
    },
    {
        "estimator__penalty": ["none"],
        "estimator__fit_intercept": [True, False],
    },
]

# -------------------------- svm
pipeline_svm = Pipeline([
    ("estimator", NuSVC(random_state=RANDOM_STATE))
])

common_params = {
    # 'C': [.01, 1.0, 10.0, 100.0, 1000.0],  # regularization
    "estimator__nu": np.arange(.1, .91, .1),  # affects number of vectors
    "estimator__shrinking": [True, False],
    "estimator__probability": [True],  # uses internal 5 fold cross validation..
    # "estimator__class_weight": [None, "balanced"],  # adjusts class imbalance
    # 'random_state': [None]
    # "estimator__class_weight": [(.55, .45), (.6, .4), (.65, .35), (.7, .3)]
}
gammas = {"estimator__gamma": ["scale", "auto"]}
coef0s = {"estimator__coef0": [.0, .5, 1.0]}
param_grid_svm = [
    # poly kernel
    {
        **{"estimator__kernel": ["poly"], "estimator__degree": [2, 3],},
        **coef0s,
        **common_params,
        **gammas,
    },
    # sigmoid kernel -- fails
    {**{"estimator__kernel": ["sigmoid"]}, **coef0s, **common_params, **gammas},
    # rbf -- fails
    {**{"estimator__kernel": ["rbf"]}, **common_params, **gammas},
    # linear
    {"estimator__kernel": ["linear"], **common_params},
]

# -------------------------- nn

pipeline_nn = Pipeline([
    ("estimator", MLPClassifier(verbose=1, max_iter=300, random_state=RANDOM_STATE))
])

param_grid_nn = {
    "estimator__hidden_layer_sizes": [(100,), (50, 50), (25, 25, 25)],
    "estimator__alpha": [10e-4, 10e-3, 10e-2, 10e-1, 1.0, 10e1, 10e2],
    "estimator__solver": ["lbfgs"],
}


# -------------------------- rf
pipeline_rf = Pipeline([
    ("estimator", RandomForestClassifier(random_state=RANDOM_STATE))
])
param_grid_rf = {
    "estimator__n_estimators": [175, 250, 325],
    "estimator__criterion": ["gini", "entropy"],
    "estimator__min_samples_split": [2, 5, 7, 10],  # influences max depth
    "estimator__max_features": ["sqrt", "log2"],  # max_features
    "estimator__min_impurity_decrease": [0.0, 0.3],
    # "estimator__class_weight": ["balanced", "balanced_subsample"],
    # "estimator__class_weight": [(.55, .45), (.6, .4), (.65, .35), (.7, .3)]
}

# -------------------------- gb

pipeline_gb = Pipeline(
    [("estimator", GradientBoostingClassifier(random_state=RANDOM_STATE))]
)
param_grid_gb = {
    "estimator__loss": ["deviance", "exponential"],  # exponential == adaboost
    "estimator__learning_rate": [0.01, 0.1],  # bate nos .1 consistentemente
    "estimator__min_samples_split": [.1, .5],  # bate nos .1 consistentemente
    "estimator__min_samples_leaf": [.1],  # bate nos .1 consistentemente
    "estimator__max_features": ["log2", "sqrt"],
    "estimator__criterion": ["friedman_mse"],
    # "estimator__subsample": [0.3, 0.5, 0.8],  # bate nos 0.8
    "estimator__subsample": [.5, 0.8],  # bate nos 0.8
    "estimator__n_estimators": [300, 400, 500],
}


estimators_collection = {
    'bayes': {
        'pipeline': pipeline_bayes,
        'param_grid':param_grid_bayes
    },
    'knn': {
        'pipeline': pipeline_knn,
        'param_grid': param_grid_knn
    },
    'logreg': {
        'pipeline': pipeline_logreg,
        'param_grid': param_grid_logreg
    },
    'svm': {
        'pipeline': pipeline_svm,
        'param_grid': param_grid_svm
    },
    'nn': {
        'pipeline': pipeline_nn,
        'param_grid': param_grid_nn
    },
    'gb': {
        'pipeline': pipeline_gb,
        'param_grid': param_grid_gb
    },
    'rf': {
        'pipeline': pipeline_rf,
        'param_grid': param_grid_rf
    }
}
