from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, precision_score,\
    recall_score
from kowalski.utils.metrics import specificity

def evaluate_on_test(estimator, test_x, test_y) -> dict:

    y_proba = estimator.predict_proba(test_x)[:, -1]
    y_pred = estimator.predict(test_x)

    return {
        'roc_auc': roc_auc_score(test_y, y_proba),
        'kappa': cohen_kappa_score(test_y, y_pred),
        'f1': f1_score(test_y, y_pred),
        'prec': precision_score(test_y, y_pred),
        'rec': recall_score(test_y, y_pred),
        'spec': specificity(test_y, y_pred)
    }
