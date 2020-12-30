from IPython.display import Image

from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score,\
    recall_score

from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import shuffle, randint

from itertools import cycle
from tqdm import tqdm

from scipy.stats import bayes_mvs, wilcoxon



import seaborn as sns

def horizontal_boxplot(data, xlabel:str=None, title:str=None):
    """
    columns in data correspond to a box
    index is ignored I think
    """

    ax = sns.boxplot(data=data, orient='h', palette='GnBu', whis=.9)
    # ax = sns.swarmplot(data=res30, orient='h', palette='magma')
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if title is not None:
        ax.set_title(title)
    # ax.figure.savefig(join(save_dir, 'boxplot.png'), dpi=500, bbox_inches='tight')
    return ax



class BNTargetShufflingAssessment:
    def __init__(
        self,
        data: pd.DataFrame,
        bayesian_network: BayesianNetwork,
        target_name: str = "response",
        n_trials=1000,
        splitter=StratifiedShuffleSplit(n_splits=30, test_size=0.2),
    ):
        self.data = data
        self.target_name = target_name
        self.x_columns = [col for col in data.columns if col != target_name]
        self.target = data[target_name]
        self.target_copy = data[target_name].copy()

        self.bn = bayesian_network
        self.bn.fit_node_states(self.data)

        self.splitter = splitter
        self.n_trials = n_trials

        self.renamer_tsa = {
            "random_luck_probas": "Random Luck Probabilities",
            "reference_aucs": "Benchmark ROC AUCs",
        }

    def run_trials(self):
        trials_results = None

        def __roc_auc_score(
            x_subset, target_name=self.target_name, max_class_suffix="pCR"
        ):
            y_proba = self.bn.predict_probability(x_subset, target_name)[
                "_".join([target_name, max_class_suffix])
            ]
            # y_predi = self.bn.predict(x_subset, target_name).applymap(lambda x: 1 if x == 'pCR' else 0)
            auc = roc_auc_score(x_subset[target_name], y_proba)  # 'micro'-averaging?
            return auc

        for split, (train_idx, test_idx) in enumerate(
            self.splitter.split(self.data[self.x_columns], self.target)
        ):

            x_train, x_test = self.data.iloc[train_idx], self.data.iloc[test_idx]

            self.bn.fit_cpds(x_train, "BayesianEstimator", "K2")
            reference_auc = __roc_auc_score(x_test)
            # _, reference_auc = roc_auc(self.bn, x_test, self.target_name)

            auc_shuffled_targets = []
            x_train_shuffled = x_train.copy()
            for trial in range(self.n_trials):

                shuffle(x_train_shuffled[self.target_name].values)  # shuffles inplace

                self.bn.fit_cpds(x_train_shuffled, "BayesianEstimator", "K2")
                auc_shuffled_target = __roc_auc_score(x_test)
                auc_shuffled_targets.append(auc_shuffled_target)

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

        self.trials_results = trials_results
        self.set_per_split_results()
        self.compute_mean_results()
        # return trials_results

    def set_per_split_results(self):
        self.random_luck_probas = self.trials_results.groupby("split").apply(
            lambda grp: grp.lucky.sum() / grp.shape[0]
        )
        self.random_luck_probas.name = "random_luck_probas"

        self.reference_aucs = pd.Series(self.trials_results["reference_auc"].unique())
        self.reference_aucs.name = "reference_aucs"

        self.per_split_results = self.random_luck_probas.to_frame().join(
            self.reference_aucs
        )

    def compute_mean_results(self):
        """
        After calling run_trials(),
        returns mean histogram and mean random luck
        """

        self.mean_random_luck_proba = self.random_luck_probas.mean()
        self.mean_roc_auc = self.reference_aucs.mean()

        print(
            f"Mean Random Luck Probability = {self.mean_random_luck_proba:.2f}\nMean ROC AUC = {self.mean_roc_auc:.2f}"
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
        # matplotlib colormaps
        # cmap = plt.get_cmap('Set2').colors
        # cmap = plt.get_cmap('tab10').colors
        # cmap = plt.get_cmap('Pastel1').colors

        # just a single color
        # cmap = ['#c47571']

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
            self.trials_results.groupby("split"), cycle(cmap)
        ):
            if ax is None:
                ax = group.random_trial_roc.plot.hist(**plot_params, color=color)
            else:
                group.random_trial_roc.plot.hist(ax=ax, **plot_params, color=color)

        ax.set_title(
            f"Target Shuffling Assessment - {self.splitter.n_splits} test sets"
        )
        ax.set_xlabel(
            f"ROC AUC\n\nAverage chance of random luck = {self.mean_random_luck_proba*100:.1f}%\n({self.n_trials} trials per test set)"
        )

        xmin, xmax = ax.get_xlim()
        hist_x_range = xmax - xmin
        ymax = ax.get_ylim()[1]

        plt.axvline(self.mean_roc_auc, color="#94c884", linestyle="dashed", linewidth=2)
        plt.text(
            self.mean_roc_auc - 0.03 * hist_x_range,
            ymax - 0.27 * ymax,
            f"AUC = {self.mean_roc_auc:.2f}",
            rotation=90,
        )

        return ax



renamer_stats = {
    'lower_bound': 'Lower Confidence Bound',
    'mean': 'Mean',
    'upper_bound': 'Upper Confidence Bound',
    'std': 'Standard Deviation',
    'wilcoxon_pvalue': 'Wilcoxon p-value on Mean'
}

renamer_metrics = {
    'roc': 'ROC AUC',
    'acc': 'Accuracy',
    'f1': 'F1',
    'prec': 'Precision',
    'rec': 'Recall'
}



class BNCrossValidationAssessment:
    def __init__(
        self,
        bayesian_network: BayesianNetwork,
        cvgen=StratifiedShuffleSplit(
            n_splits=30, test_size=0.2, random_state=randint(2 ** 32 - 1)
        ),
    ):
        """Automates Cross-Validation of a Bayesian Network

        Args:
            estimator (BayesianNetwork)
            cvgen (sklearn splitter): Defaults to StratifiedShuffleSplit( n_splits=30, test_size=0.2, random_state=randint(2 ** 32 - 1) ).
        """        
        # a Bayesian network
        self.bn = bayesian_network
        self.cvgen = cvgen


    def run(self, x:pd.DataFrame, y):

        res = {'roc':[], 'acc':[], 'f1':[], 'prec':[], 'rec':[]}

        self.bn.fit_node_states(x)

        for train_idx, test_idx in tqdm(self.cvgen.split(x, y)):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            # _, y_test = y.iloc[train_idx], y.iloc[test_idx]
            _, y_test = y[train_idx], y[test_idx]
            
            assert type(x_train) == pd.DataFrame
            self.bn.fit_cpds(x_train, 'BayesianEstimator', 'K2')

            y_proba = self.bn.predict_probability(x_test, 'response')['response_pCR']
            y_predi = self.bn.predict(x_test, 'response').applymap(
                lambda x: 1 if x == 'pCR' else 0
            )
            
            res['roc'].append(roc_auc_score(y_test, y_proba))
            res['acc'].append(accuracy_score(y_test, y_predi))
            res['f1'].append(f1_score(y_test, y_predi))
            res['prec'].append(precision_score(y_test, y_predi))
            res['rec'].append(recall_score(y_test, y_predi))

        res = pd.DataFrame(res)
        res.index.name = 'split'
        self.res_ = res


    def aggregate_results(self):

        def lower_bound(series):
            return bayes_mvs(series, alpha=.95)[0].minmax[0]
            
        def upper_bound(series):
            return bayes_mvs(series, alpha=.95)[0].minmax[1]

        def wilcoxon_pvalue(series):
            return wilcoxon(series-series.mean()).pvalue

        return self.res_.agg([
            lower_bound, 'mean', upper_bound, 'std', wilcoxon_pvalue
        ]).rename(index=renamer_stats, columns=renamer_metrics)


    def plot_results(self):
        ax = horizontal_boxplot(
            data=self.res_.rename(columns=renamer_metrics),
            xlabel=f'{self.cvgen.n_splits} splits @ [test size]={self.cvgen.test_size}',
            title='BayesianNetwork'
        )
        return ax


from SALib.sample import saltelli
from SALib.analyze import sobol
from sklearn.preprocessing import OrdinalEncoder

class BNSensitivityAnalysis:


    def __init__(self, bayesian_network: BayesianNetwork, dataset: pd.DataFrame):
        self.df = dataset
        self.bn = bayesian_network.fit_node_states(dataset).fit_cpds(dataset)

        self.problem = {
            'num_vars': self.df.shape[1],
            'names': list(self.df.columns),
            'bounds': [[]]
        }

        self.sampler = saltelli
        self.analyzer = sobol


    def run_unsupervised_simulation(self, N=1000):
        self.N = N
        try:
            return self.S1T, self.S2
        except NameError:
            # If calc_second_order is True, the resulting matrix has N * (2D + 2) rows.        
            encoder = OrdinalEncoder().fit(self.df)
            sample = self.sampler.sample(problem=self.problem, N=N)
            sample = pd.DataFrame(
                encoder.inverse_transform(np.rint(sample)),
                columns=self.df.columns
            )
            # WARNING: hard coded!!
            predictions = self.bn.predict(sample, 'response').applymap(
                lambda x: 1 if x == 'pCR' else 0
            )

            si = self.analyzer.analyze(problem, sample)
            self.S1T = pd.DataFrame(
                {'S1': si['S1'], 'ST':si['ST']},
                index=self.df.columns
            )
            self.S2 = pd.DataFrame(
                si['S2'], index=self.df.columns, columns=self.df.columns
            )
            return self.S1T, self.S2


    def run_supervised(self):
        pass