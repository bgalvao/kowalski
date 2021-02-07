from sklearn.naive_bayes import GaussianNB
from sklearn.svm import NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest

from imblearn.pipeline import make_pipeline
import numpy as np

RANDOM_STATE = None

# -------------------------- bayes
pipeline_bayes = make_pipeline(GaussianNB())
param_grid_bayes = {"gaussiannb__var_smoothing": [0.05, 0.1, 0.15, 0.2, 0.25]}


# -------------------------- knn
pipeline_knn = make_pipeline(KNeighborsClassifier(metric="euclidean"))
param_grid_knn = {
    "kneighborsclassifier__n_neighbors": [2, 3, 4, 5, 6, 7],
    "kneighborsclassifier__weights": ["uniform", "distance"],
    "kneighborsclassifier__p": [1, 2, 3],  # parameter p of minkowski distance
}


# -------------------------- log reg
pipeline_logreg = make_pipeline(
    LogisticRegression(max_iter=50000, random_state=RANDOM_STATE, solver='saga')
)

param_grid_logreg = {
    "logisticregression__penalty": ["elasticnet"],
    "logisticregression__l1_ratio": np.arange(0.0, 1.1, 0.1),  # i.e. spectrum from l2 to l1
    "logisticregression__fit_intercept": [True, False],
}

# -------------------------- svm
pipeline_svm = make_pipeline(NuSVC(random_state=RANDOM_STATE))

common_params = {
    "nusvc__nu": [.3, .5, .8],  # affects number of vectors
    "nusvc__shrinking": [True, False],
    "nusvc__probability": [True],  # uses internal 5 fold cross validation..
}
gammas = {"nusvc__gamma": ["scale", "auto"]}
coef0s = {"nusvc__coef0": [.0, .5, 1.0]}
param_grid_svm = [
    # poly kernel
    {
        **{"nusvc__kernel": ["poly"], "nusvc__degree": [2, 3],},
        **coef0s,
        **common_params,
        **gammas,
    },
    # sigmoid kernel -- fails
    {**{"nusvc__kernel": ["sigmoid"]}, **coef0s, **common_params, **gammas},
    # rbf -- fails
    {**{"nusvc__kernel": ["rbf"]}, **common_params, **gammas},
    # linear
    {"nusvc__kernel": ["linear"], **common_params},
]

# -------------------------- nn

pipeline_nn = make_pipeline(MLPClassifier(verbose=1, max_iter=300, random_state=RANDOM_STATE))

param_grid_nn = {
    "mlpclassifier__hidden_layer_sizes": [(20,), (20, 20), (20, 20, 20)],
    "mlpclassifier__alpha": [10e-3, 10e-2, 10e-1],
    "mlpclassifier__solver": ["lbfgs"],
}


# -------------------------- rf
pipeline_rf = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE))
param_grid_rf = {
    "randomforestclassifier__n_estimators": [175, 250, 325],
    "randomforestclassifier__criterion": ["gini", "entropy"],
    "randomforestclassifier__min_samples_split": [2, 5, 7, 10],  # influences max depth
    "randomforestclassifier__max_features": ["sqrt", "log2"],  # max_features
    "randomforestclassifier__min_impurity_decrease": [0.0, 0.3],
    # "estimator__class_weight": ["balanced", "balanced_subsample"],
}

# -------------------------- gb

pipeline_gb = make_pipeline(GradientBoostingClassifier(random_state=RANDOM_STATE))
param_grid_gb = {
    "gradientboostingclassifier__loss": ["deviance", "exponential"],  # exponential == adaboost
    "gradientboostingclassifier__learning_rate": [0.01, 0.1],  # bate nos .1 consistentemente
    "gradientboostingclassifier__min_samples_split": [.1, .5],  # bate nos .1 consistentemente
    "gradientboostingclassifier__min_samples_leaf": [.1],  # bate nos .1 consistentemente
    "gradientboostingclassifier__max_features": ["log2", "sqrt"],
    "gradientboostingclassifier__criterion": ["friedman_mse"],
    "gradientboostingclassifier__subsample": [1.0, 0.8],  # bate nos 0.8
    "gradientboostingclassifier__n_estimators": [200, 300, 400],
}


# -------------------------- genetic programming

from gplearn.genetic import SymbolicClassifier
from gplearn.functions import make_function

# https://gplearn.readthedocs.io/en/stable/reference.html#symbolic-classifier
sc = SymbolicClassifier(
    population_size=2000,
    generations=20, 
    tournament_size=25, 
    const_range=(-1.5, 1.5), 
#     init_depth=(10, 20), 
#     init_method='full',
    init_method='half and half',
    function_set=(
        'add', 'sub', 'mul', 'div', 'cos', 'log'
#         'sin', 'min', 'max', 'sqrt',  #'neg', 'tan'
    ), 
    transformer='sigmoid',
    
#     metric=mf_wf, stopping_criteria=2.0,
    parsimony_coefficient=0.0001,
    
    p_crossover=0.7,
    p_subtree_mutation=0.2,
    
    p_hoist_mutation=0.00,   
    p_point_mutation=0.1,
    p_point_replace=0.05,
    
    max_samples=.9,
#     feature_names=train_x.columns,

    low_memory=True,
    n_jobs=-1,
    verbose=1,
    random_state=None
)
pipeline_gp = make_pipeline(sc)
param_grid_gp = {}

estimators_collection = {
    'nb': (pipeline_bayes, param_grid_bayes),
    'knn': (pipeline_knn, param_grid_knn),
    'lr': (pipeline_logreg, param_grid_logreg),
    'svm': (pipeline_svm, param_grid_svm),
    'mlp': (pipeline_nn, param_grid_nn),
    'gbc': (pipeline_gb, param_grid_gb),
    'rf': (pipeline_rf, param_grid_rf),
    'gp': (pipeline_gp, param_grid_gp)
}
