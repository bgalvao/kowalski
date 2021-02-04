from itertools import cycle
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from scipy.stats import norm


class TargetShufflingAssessment:
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

        self.estimator = deepcopy(estimator)

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



