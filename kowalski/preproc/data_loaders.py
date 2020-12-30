import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def _find_dataset_directory():
    base_path = './dataset'
    while not os.path.exists(base_path):
        base_path = os.path.join('..', base_path)
    return base_path


def clean_df(filename="phillips.csv", scale_data=True):

    """
    Reads and cleans the source data file.
    Returns values as they are.

    filename : a file in the dataset directory
    """

    filepath = os.path.join(
        _find_dataset_directory(),
        filename
    )

    global numerical_features, categorical_features, ordinal_features, target, \
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

    numerical_features = ["Age"] + radiomic_features
    categorical_features = ["ovarian_status", "histology", "subtypes"]
    ordinal_features = ["stage", "biopsy_grade", "clinical_nodal_status"]

    df = df.dropna(); df['biopsy_grade'] = df['biopsy_grade'].astype(str)
    # df["biopsy_grade"] = df["biopsy_grade"].fillna(2.0).astype(str)  # mode

    df = df[[target] + categorical_features + ordinal_features + numerical_features]
    df.rename(columns={'[target]response':'response'}, inplace=True)
    target = 'response'

    if scale_data:
        df[numerical_features] = StandardScaler().fit_transform(df[numerical_features])

    return df



def _preproc_target(df):
    """
    Returns target variable (binary) and the LabelBinarizer instance
    that created it.
    """
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(df[target]).ravel()
    y = pd.Series(y, index=df.index, name='response')
    print(label_binarizer.classes_)
    return y, label_binarizer





def load_simple_format():
    """
    all the xy you could ask for
    """
    df = clean_df()
    y, label_binarizer = _preproc_target(df)
    x = df.iloc[:, 1:]
    return x, y, label_binarizer




def load_sklearn_format():
    df = clean_df()

    ordinal_encoder = OrdinalEncoder()
    ordinal_data = pd.DataFrame(
        data=ordinal_encoder.fit_transform(df[ordinal_features]),
        columns=ordinal_features,
        index=df.index
    ).astype(np.uint8)
    
    onehot_encoder = OneHotEncoder(
        drop=['Premenopausal', 'Ductal', 'ER-/HER-'],
        sparse=False
    )
    onehot_encoder.fit(df[categorical_features])
    categorical_data = pd.DataFrame(
        data=onehot_encoder.transform(df[categorical_features]),
        columns=onehot_encoder.get_feature_names(categorical_features),
        index=df.index
    ).astype(np.uint8)

    x = categorical_data.join(ordinal_data).join(df[numerical_features])
    y, label_binarizer = _preproc_target(df)
    return x, y, label_binarizer





def load_causalnex_format():

    
    def discretize_numerical_features(df_num:pd.DataFrame, k_bins:int=6):
        """
        - Discretizes some numerical subset of df, df_num.
        - Acceptable values for k_bins is 6, 4, 3, 2.
        - This function standardizes data before discretizing with 1-dimensional kmeans.
        """
        assert k_bins in {2, 3, 4, 6}
        def default_discretizer(k_bins=k_bins):
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
                    n_bins=k_bins, strategy='kmeans', encode='ordinal'
                )]
            ])

            return pipe, bin_labels
        
        pipe, bin_labels = default_discretizer(k_bins)
        pipe.fit(df_num)

        res = pd.DataFrame(
            pipe.transform(df_num),
            columns=df_num.columns,
            index=df_num.index
        )

        return res, pipe, bin_labels



def save(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def read(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)




rename_dict_metrics = {
    'roc': 'ROC AUC',
    'acc': 'Accuracy',
    'f1': 'F1',
    'prec': 'Precision',
    'rec': 'Recall'
}