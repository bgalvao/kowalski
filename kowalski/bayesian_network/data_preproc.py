import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OrdinalEncoder,\
    LabelBinarizer
from sklearn.pipeline import Pipeline

import pickle

def clean_df(filepath="../../dataset/phillips.csv"):

    """
    Reads and cleans the source data file.
    Returns values as they are.
    """

    global num_features, cat_features, ord_features, target, \
        radiomic_features, non_imaging_features

    # clean and reorder data
    df = pd.read_csv(filepath, index_col="ID").drop(
        ["Label", "Label_ax", "biological_subtypes_new"], axis=1
    )

    target = df.columns[7]
    assert target == "[target]response"
    non_imaging_features = list(df.columns[:7])
    radiomic_features = [
        col
        for col in df.columns
        if col not in non_imaging_features and target not in col
    ]

    num_features = ["Age"] + radiomic_features
    cat_features = ["ovarian_status", "histology", "subtypes"]
    ord_features = ["stage", "biopsy_grade", "clinical_nodal_status"]

    df = df.dropna(); df['biopsy_grade'] = df['biopsy_grade'].astype(str)
    # df["biopsy_grade"] = df["biopsy_grade"].fillna(2.0).astype(str)  # mode


    df = df[[target] + cat_features + ord_features + num_features]

    return df




def preproc_target(df):
    """
    Returns target variable (binary) and the LabelBinarizer instance
    that created it.
    """
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(df[target]).ravel()
    y = pd.Series(y, index=df.index, name='response')
    print(label_binarizer.classes_)
    return y, label_binarizer




def discretize_numerical_features(df_num, k_bins=6, strategy='uniform'):
    """
    - Discretizes some numerical subset of df, df_num.
    - Acceptable values for k_bins is 6, 4, 3, 2.
    - This function standardizes data before discretizing with 1-dimensional kmeans.
    """
    assert k_bins in {2, 3, 4, 6}
    assert strategy in {'uniform', 'quantile', 'kmeans'}

    def default_discretizer(k_bins=k_bins, strategy=strategy):
        bin_labels = [
            'extreme-low', 'low', 'mid-low', 'mid-high', 'high', 'extreme-high'
        ]
        if k_bins == 4:
            bin_labels = bin_labels[1:-1]
        elif k_bins == 3:
            bin_labels = ['low', 'mid', 'high']
        elif k_bins == 2:
            bin_labels = ['low', 'high']
        
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ['discretize', KBinsDiscretizer(
                n_bins=k_bins, strategy=strategy, encode='ordinal'
            )]
        ])

        return pipe, bin_labels
    
    pipe, bin_labels = default_discretizer(k_bins)
    pipe.fit(df_num)

    res = pd.DataFrame(
        pipe.transform(df_num),
        columns=df_num.columns,
        index=df_num.index
    ).astype(np.uint8)

    return res, pipe, bin_labels




def encode_nominal_features(df_nom):
    """
    """

    ord_encoder = OrdinalEncoder().fit(df_nom)
    ord_encoded_features = ord_encoder.transform(df_nom)

    res = pd.DataFrame(
        ord_encoded_features, index=df_nom.index, columns=df_nom.columns
    )

    return res, ord_encoder  # ord_encoder stores label info in categories_




def read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save(obj, filepath):
    with open(filepath, 'rw') as f:
        pickle.dump(obj, f)



rename_dict_metrics = {
    'roc': 'ROC AUC',
    'acc': 'Accuracy',
    'f1': 'F1',
    'prec': 'Precision',
    'rec': 'Recall'
}