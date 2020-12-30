from imblearn.pipeline import Pipeline


from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.svm import LinearSVC


from imblearn.over_sampling import SMOTENC, KMeansSMOTE
import numpy as np


def prepend2pipeline(step: tuple, pipeline: Pipeline) -> Pipeline:
    if type(pipeline) == Pipeline:
        return Pipeline([step] + pipeline.get_params()["steps"])
    elif type(pipeline) == list:
        return Pipeline([step] + pipeline)
    else:
        raise TypeError("`pipeline` has to be either a list or a Pipeline")

#

"""
------------------------------------------------------------------ Feature selection
"""


feature_selection = FeatureUnion(
    transformer_list=[
        (
            "clinical",
            SelectFromModel(
                max_features=13, estimator=LinearSVC(C=0.1, penalty="l1", dual=False)
            ),
        ),
        ("radiomics", SelectKBest(score_func=mutual_info_classif)),
    ]
)

radiomics_k_range = {"feature_selection__radiomics__k": np.arange(50, 201, 50)}


def introduce_feature_selection(pipeline, param_grid={}):
    new_pipe = prepend2pipeline(("feature_selection", feature_selection), pipeline)
    new_param_grid = prepend2param_grid(radiomics_k_range, param_grid)
    return (new_pipe, new_param_grid)


"""
------------------------------------------------------ Oversample
"""
oversample = ("oversample", ())
# oversample = ('oversample', KMeansSMOTE())
def introduce_oversample(pipeline) -> Pipeline:
    return prepend2pipeline(oversample, pipeline)
