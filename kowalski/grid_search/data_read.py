import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    LabelBinarizer,
)


def clean_df(
    filepath="../dataset/csv/phillips.csv",
    prev_selected_file="../dataset/csv/orange/last_feature_selection.csv",
):
    global num_features, cat_features, ord_features, prev_sel, target, \
        radiomic_features
    prev_sel = None

    # clean and reorder data
    df = pd.read_csv(filepath, index_col="ID").drop(
        ["Label", "Label_ax", "biological_subtypes_new"], axis=1
    )

    target = df.columns[7]
    assert target == '[target]response'
    clinical_features = list(df.columns[:7])
    radiomic_features = [
        col for col in df.columns if col not in clinical_features and target not in col
    ]

    if prev_selected_file is not None:
        prev_sel = [
            col
            for col in pd.read_csv(prev_selected_file, index_col="ID").columns
            if col not in clinical_features and "[target]" not in col
        ]

    num_features = ["Age"] + radiomic_features
    cat_features = ["ovarian_status", "histology", "subtypes"]
    ord_features = ["stage", "biopsy_grade", "clinical_nodal_status"]

    df["biopsy_grade"] = df["biopsy_grade"].fillna(2.0)  # mode

    df = df[[target] + cat_features + ord_features + num_features]

    return df


def sklearn_xy(df, drop_base_categories=True, all_categorical=False):
    global lby, cats, ohe


    nums = pd.DataFrame(
        StandardScaler().fit_transform(df[num_features]),
        columns=num_features,
        index=df.index,
    )

    if all_categorical:
        # treat even ordinal features as categorical
        nominal_features = cat_features + ord_features
        ohe = OneHotEncoder(sparse=False)
        noms = ohe.fit_transform(df[nominal_features])
        noms = pd.DataFrame(
            noms, columns=ohe.get_feature_names(nominal_features), index=df.index
        )
        
        x = noms.join(nums)
        if prev_sel is not None:
            x = x[list(noms.columns) + ['Age'] + prev_sel]

    else:
        # ohe = OneHotEncoder(drop="first", sparse=False)
        if drop_base_categories:
            ohe = OneHotEncoder(drop=['Premenopausal', 'Ductal', 'ER-/HER-'], sparse=False)
        else:
            ohe = OneHotEncoder(sparse=False)
        cats = ohe.fit_transform(df[cat_features])

        cats = pd.DataFrame(
            cats, columns=ohe.get_feature_names(cat_features), index=df.index
        )

        ords = pd.DataFrame(
            OrdinalEncoder().fit_transform(df[ord_features]),
            columns=ord_features,
            index=df.index,
        )

        x = cats.join(ords).join(nums)
        if prev_sel is not None:
            x = x[list(cats.columns) + list(ords.columns) + ["Age"] + prev_sel]



    lb = LabelBinarizer()
    y = lb.fit_transform(df[target]).ravel()
    print(lb.classes_)
    return x, y


if __name__ == '__main__':
    df = clean_df()
    x, y = sklearn_xy(df, drop_base_categories=False)
