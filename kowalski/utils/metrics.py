
from sklearn.metrics import make_scorer, cohen_kappa_score, confusion_matrix,\
    matthews_corrcoef

# https://uberpython.wordpress.com/2012/01/01/precision-recall-sensitivity-and-specificity/
def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    denominator = tn + fp
    if denominator == 0:
        return 0
    else:
        return tn / denominator

mcc_score = make_scorer(matthews_corrcoef)
kappa_score = make_scorer(cohen_kappa_score)
specificity_score = make_scorer(specificity)

scorers = {
    'roc_auc': 'roc_auc',
    'kappa': kappa_score,
    # 'mcc': mcc_score,
    # 'accuracy': 'balanced_accuracy',
    'f1': 'f1',
    'prec': 'precision',
    'rec': 'recall',
    'spec': specificity_score,
}

rename_dict_metric = {
    'roc_auc': 'ROC AUC',
    'acc': 'Accuracy',
    'f1': 'F1',
    'prec': 'Precision',
    'acc': 'Accuracy',
    'rec': 'Recall',
    'spec': 'Specificity',
    'kappa': 'Kappa',
    'mcc': 'MCC'
}