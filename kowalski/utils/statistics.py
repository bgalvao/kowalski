from scipy.stats import bayes_mvs, wilcoxon, t, sem, norm
import numpy as np


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def lcb(series):
    return bayes_mvs(series, .95)[0].minmax[0]

def ucb(series):
    return bayes_mvs(series, .95)[0].minmax[1]

def wilcoxon_pvalue(series):
    try:
        return wilcoxon(series-series.mean()).pvalue
    except ValueError as error:
        if 'x - y is zero for all elements' in str(error):
            return 1.0
        else:
            raise error


def delong_test(y_true, y_proba_a, y_proba_b):
    # https://biasedml.com/roc-comparison/
    # all the credit goes to Laksan Nathan. For now, I just plugged his code here

    def auc(X, Y):
        return 1/(len(X)*len(Y)) * sum([kernel(x, y) for x in X for y in Y])

    def kernel(X, Y):
        return .5 if Y==X else int(Y < X)

    def structural_components(X, Y):
        V10 = [1/len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([kernel(x, y) for x in X]) for y in Y]
        return V10, V01
        
    def get_S_entry(V_A, V_B, auc_A, auc_B):
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])

    def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB)**(.5))

    def group_preds_by_label(proba, actual):
        X = [p for (p, a) in zip(proba, actual) if a]
        Y = [p for (p, a) in zip(proba, actual) if not a]
        return X, Y
    
    X_A, Y_A = group_preds_by_label(y_proba_a, y_true)
    X_B, Y_B = group_preds_by_label(y_proba_b, y_true)
    V_A10, V_A01 = structural_components(X_A, Y_A)
    V_B10, V_B01 = structural_components(X_B, Y_B)

    auc_A = auc(X_A, Y_A)
    auc_B = auc(X_B, Y_B)
    
    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
            + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
            + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
    
    # Two tailed test
    z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = norm.sf(abs(z))*2
    return z, p