from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import ipywidgets as iw



def plot_outliers(df, outlier_model):
    
    x1 = iw.Dropdown(df.columns)
    x2 = iw.Dropdown(df.columns)

    outlier_model_params = outlier_model.get_params()
