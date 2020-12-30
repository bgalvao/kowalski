from itertools import cycle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

class BBTargetShufflingAssessment:
    def __init__(
        self,
        estimator,
        X: pd.DataFrame,
        Y,
        external_X=None,
        external_Y=None,
        n_trials=1000,
        splitter=StratifiedShuffleSplit(n_splits=10, test_size=0.2),
    ):

        self.estimator = estimator

        self.x = X
        self.y = Y

        self.external_x = external_X
        self.external_y = external_Y

        self.splitter = splitter
        self.n_trials = n_trials

        self.renamer_tsa = {
            "random_luck_probas": "Random Luck Probabilities",
            "reference_aucs": "Benchmark ROC AUCs",
        }


    def run_trials(self):

        trials_results = None

        def __roc_auc_score(x_test, y_test):
            # MIND YOU: this y_proba is hardcoded for binary classification
            y_proba = self.estimator.predict_proba(x_test)[:, -1]
            return roc_auc_score(y_test, y_proba)

        if self.external_x is not None:

            assert self.external_y is not None

            self.estimator.fit(self.x, self.y)
            reference_auc = __roc_auc_score(self.external_x, self.external_y)

            auc_shuffled_targets = []
            y_shuffled = self.y.copy()

            for _ in range(self.n_trials):
                np.random.shuffle(y_shuffled)
                self.estimator.fit(self.x, y_shuffled)
                auc_shuffled_targets.append(__roc_auc_score(
                    self.external_x,
                    self.external_y
                ))

            subresult = pd.DataFrame(
                data={
                    "random_trial_roc": auc_shuffled_targets,
                    "trial": range(1, self.n_trials + 1),
                },
            ).set_index("trial")

            subresult["reference_auc"] = reference_auc
            subresult["split"] = 'vs. external'
            subresult["lucky"] = subresult["random_trial_roc"] > reference_auc
            subresult = subresult.reset_index().set_index(["split", "trial"])

            trials_results = (
                trials_results.append(subresult)
                if trials_results is not None
                else subresult
            )

        else:

            for split, (train_idx, test_idx) in enumerate(
                self.splitter.split(self.x, self.y)
            ):

                x_train, x_test = self.x.iloc[train_idx], self.x.iloc[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]

                # for a train/test split, get the regular evaluation
                self.estimator.fit(x_train, y_train)
                reference_auc = __roc_auc_score(x_test, y_test)

                # then run n_trials with shuffled target training
                auc_shuffled_targets = []
                y_train_shuffled = y_train.copy()
                for _ in range(self.n_trials):
                    np.random.shuffle(y_train_shuffled)
                    self.estimator.fit(x_train, y_train_shuffled)
                    auc_shuffled_targets.append(__roc_auc_score(x_test, y_test))

                subresult = pd.DataFrame(
                    data={
                        "random_trial_roc": auc_shuffled_targets,
                        "trial": range(1, self.n_trials + 1),
                    },
                ).set_index("trial")

                subresult["reference_auc"] = reference_auc
                subresult["split"] = split + 1
                subresult["lucky"] = subresult["random_trial_roc"] > reference_auc
                subresult = subresult.reset_index().set_index(["split", "trial"])

                trials_results = (
                    trials_results.append(subresult)
                    if trials_results is not None
                    else subresult
                )

        self.trials_results_ = trials_results
        self._set_per_split_results()
        self._compute_mean_results()
        # return trials_results

    def _set_per_split_results(self):
        self.random_luck_probas_ = self.trials_results_.groupby("split").apply(
            lambda grp: grp.lucky.sum() / grp.shape[0]
        )
        self.random_luck_probas_.name = "random_luck_probas"

        self.reference_aucs_ = pd.Series(self.trials_results_["reference_auc"].unique())
        self.reference_aucs_.name = "reference_aucs"

        self.per_split_results = self.random_luck_probas_.to_frame().join(
            self.reference_aucs_
        )

    def _compute_mean_results(self):
        """
        After calling run_trials(),
        returns mean histogram and mean random luck
        """

        self.mean_random_luck_proba_ = self.random_luck_probas_.mean()
        self.mean_roc_auc__ = self.reference_aucs_.mean()

        print(
            f"Mean Random Luck Probability = {self.mean_random_luck_proba_:.2f}\nMean ROC AUC = {self.mean_roc_auc__:.2f}"
        )

    def boxplot_results(self):
        plotting_data = self.per_split_results.rename(
            columns={
                "random_luck_probas": "Random Luck Probabilities",
                "reference_aucs": "Benchmark ROC AUCs",
            }
        )
        fig, ax = plt.subplots(figsize=(5.9, 1.7))

        ax = sns.boxplot(
            data=plotting_data, orient="h", palette="GnBu", whis=0.9, ax=ax
        )

        ax.set_xlabel(
            f"\n(#trials = {self.n_trials}, #crossvalidation-folds = {self.splitter.n_splits})"
        )
        return ax


    def histogram(self):

        # color map from colorbrewer
        cmap = [
            "#8dd3c7",
            "#ffffb3",
            "#bebada",
            "#fb8072",
            "#80b1d3",
            "#fdb462",
            "#b3de69",
            "#fccde5",
            "#d9d9d9",
            "#bc80bd",
        ]

        plot_params = {"alpha": 0.7, "bins": 21}

        ax = None
        for (name, group), color in zip(
            self.trials_results_.groupby("split"), cycle(cmap)
        ):
            if ax is None:
                ax = group.random_trial_roc.plot.hist(**plot_params, color=color)
            else:
                group.random_trial_roc.plot.hist(ax=ax, **plot_params, color=color)

        xmin, xmax = ax.get_xlim()
        hist_x_range = xmax - xmin
        ymax = ax.get_ylim()[1]

        rename_dict_estimator = {
            'LogisticRegression': 'Logistic Regression',
            'MLPClassifier': 'Neural Network',
            'NuSVC': 'SVM',
            'GradientBoostingClassifier': 'Gradient Boosting',
            'KNeighborsClassifier': 'K-Nearest Neighbors',
            'GaussianNB': 'Naive Bayes'
        }

        if self.external_x is None:
            ax.set_title(
                f"Target Shuffling Assessment - {self.splitter.n_splits} bootstrapped test sets"
            )
            ax.set_xlabel(
                f"ROC AUC\n\nAverage chance of random luck = {self.mean_random_luck_proba_*100:.1f}%\n({self.n_trials} trials per test set)"
            )
            plt.axvline(self.mean_roc_auc__, color="#94c884", linestyle="dashed", linewidth=2)
            plt.text(
                self.mean_roc_auc__ - 0.03 * hist_x_range,
                ymax - 0.39 * ymax,
                f"Avg. AUC = {self.mean_roc_auc__:.2f}",
                rotation=90,
            )
        
        else:
            estimator_name = self.estimator.steps[-1][-1].__str__().split('(')[0]
            estimator_name = rename_dict_estimator[estimator_name]
            ax.set_title(
                f"Target Shuffling Assessment - vs. external test set\n{estimator_name}"
            )
            ax.set_xlabel(
                f"ROC AUC\n\nProbability of random luck = {self.mean_random_luck_proba_*100:.1f}%\n({self.n_trials} trials)"
            )
            plt.axvline(self.mean_roc_auc__, color="#94c884", linestyle="dashed", linewidth=2)
            plt.text(
                self.mean_roc_auc__ - 0.03 * hist_x_range,
                ymax - 0.29 * ymax,
                f"AUC = {self.mean_roc_auc__:.2f}",
                rotation=90,
            )


        return ax




def delong_test(y_true, y_proba_a, y_proba_b):
    # https://biasedml.com/roc-comparison/
    # all the credit goes to Laksan Nathan. I just plugged code here.

    # Model A (random) vs. "good" model B
    y_proba_a = np.array([.5, .5, .5, .5, .5, .5, .5, .5, .5, .5])
    y_proba_b = np.array([.2, .5, .1, .4, .9, .8, .7, .5, .9, .8])

    y_true= np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    def group_y_proba_by_label(y_true, y_proba):
        X = [p for (p, a) in zip(y_proba, y_true) if a]
        Y = [p for (p, a) in zip(y_proba, y_true) if not a]
        return X, Y

    X_A, Y_A = group_y_proba_by_label(y_true, y_proba_a)
    X_B, Y_B = group_y_proba_by_label(y_true, y_proba_b)


    def structural_components(X, Y):
        V10 = [1/len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
        V01 = [1/len(X) * sum([kernel(x, y) for x in X]) for y in Y]
        return V10, V01

    V_A10, V_A01 = structural_components(X_A, Y_A)
    V_B10, V_B01 = structural_components(X_B, Y_B)

    def auc(X, Y):
        def kernel(X, Y):
            return .5 if Y==X else int(Y < X)
        return 1/(len(X)*len(Y)) * sum([kernel(x, y) for x in X for y in Y])

    auc_A = auc(X_A, Y_A)
    auc_B = auc(X_B, Y_B)

    # Compute entries of covariance matrix S (covar_AB = covar_BA)

    def get_S_entry(V_A, V_B, auc_A, auc_B):
        return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])

    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
            + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
            + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))

    # Two tailed test
    def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB)**(.5))
    
    z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
    p = st.norm.sf(abs(z))*2
    return z, p


    

    


        
    


