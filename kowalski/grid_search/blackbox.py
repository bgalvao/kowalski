"""
Copyright Bernardo Galvao 2020 github@bgalvao
email: bernardo.galvao.ml@gmail.com
"""

print('(re)loaded')

from tempfile import mkdtemp
from shutil import rmtree
from copy import deepcopy
from time import perf_counter as time_split
from datetime import timedelta
import os
import itertools
import json


from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('darkgrid')

from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold,\
    GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, \
    precision_score, recall_score, cohen_kappa_score, roc_curve, matthews_corrcoef
from sklearn.metrics import confusion_matrix

import mlflow

from scrpts.grid_search.estimator_pipelines import estimators_collection as ec
from scrpts.grid_search.assessment import BBTargetShufflingAssessment
from scrpts.utils import *

def save_plot(ax, path):
    ax.figure.savefig(path, dpi=500, bbox_inches='tight')


# hard coded for binary classification
class BlackBoxClassifierSearch:

    def __init__(
        self,
        bootstrap_test_gen=StratifiedShuffleSplit(
            n_splits=10, test_size=.2, random_state=np.random.randint(2**32-1)
        ),
        cvgen=StratifiedKFold(n_splits=5, shuffle=True),
        fit_criterium='roc_auc',
        pipeline_prepends=None, param_grid_preprends=None,
        sample_weight_allocator=None
    ):
        """[summary]

        Args:
            bootstrap_test_gen ([type], optional): [description]. Defaults to StratifiedShuffleSplit( n_splits=10, test_size=.2, random_state=np.random.randint(2**32-1) ).
            cvgen ([type], optional): [description]. Defaults to StratifiedKFold(n_splits=5, shuffle=True).
            fit_criterium (str, optional): [description]. Defaults to 'roc_auc'.
            pipeline_prepends ([type], optional): [description]. Defaults to None.
            param_grid_preprends ([type], optional): [description]. Defaults to None.
            sample_weight_allocator ([type], optional): [description]. Defaults to None.
        """
        self.bootstrap_test_gen = bootstrap_test_gen
        self.cvgen = cvgen

        self.fit_criterium = fit_criterium

        self.pipeline_prepends = pipeline_prepends
        self.param_grid_prepends = param_grid_preprends

        self.sample_weight_allocator = sample_weight_allocator


    def _register_feature_set(self, x:pd.DataFrame, save_dir):
        try:
            if self.feature_set == list(x.columns):
                return self
            else:
                raise AttributeError('Feature set of passed x does not match registry!')
        except AttributeError:
            # print(f'Registering feature set of the {sa
            # self.__repr__().split(".")[-1].strip(">")}')
            self.feature_set = list(x.columns)
            with open('feature_set.json', 'w') as f:
                data = json.dumps(self.feature_set, indent=2)
                f.write(data)
        return self


    def _prepare_pipeline(self, pipeline: Pipeline, param_grid):
        if self.pipeline_prepends is None:
            return pipeline, param_grid
        
        modified_pipeline = Pipeline(self.pipeline_prepends.steps + pipeline.steps)

        if self.param_grid_prepends is not None:
            if type(self.param_grid_prepends) == dict:
                if type(param_grid) == list:
                    modified_param_grid = [
                        {**self.param_grid_prepends, **subgrid}
                        for subgrid in param_grid
                    ]
                    return modified_pipeline, modified_param_grid
                elif type(param_grid) == dict:
                    modified_param_grid = {**self.param_grid_prepends, **param_grid}
                    return modified_pipeline, modified_param_grid
                else:
                    raise TypeError('param_grid should be either a list of dicts or a dict')
            
            elif type(self.param_grid_prepends) == list:
                if type(param_grid) == list:
                    modified_param_grid = [
                        {**pgp, **pg}
                        for pgp, pg in itertools.product(
                            self.param_grid_prepends, param_grid
                        )
                    ]
                    return modified_pipeline, modified_param_grid
                elif type(param_grid) == dict:
                    modified_param_grid = [{**pgp, **param_grid} for pgp in self.param_grid_prepends]
                    return modified_pipeline, modified_param_grid
                else:
                    raise TypeError('param_grid should be either a list of dicts or a dict')
        
        return modified_pipeline, param_grid


    def fit_with_bootstrapped_test_sets(self, X:pd.DataFrame, Y, save_dir, sample_weights=None):
        """
        This cannot be run before .fit.

        Args:
            X (pd.DataFrame): [description]
            Y ([type]): [description]
            save_dir ([type]): [description]

        Returns:
            self
        """
        
        self._register_feature_set(X, save_dir)
        self.bootstrap_fits_ = {estimator:[] for estimator in self.fits_.keys()}
        
        mlflow.sklearn.autolog()

        for i, (train_idx, _) in tqdm(
            enumerate(self.bootstrap_test_gen.split(X, Y))
        ):

            x_train, y_train = X.iloc[train_idx], Y[train_idx]
            
            for estimator, gs in self.fits_.items():

                bestimator = deepcopy(gs.best_estimator_)

                cachedir = mkdtemp()
                bestimator.set_params(memory=cachedir)
                with mlflow.start_run(run_name=f'{estimator}_bs'):
                    fit_params = {'estimator__sample_weight': sample_weights[train_idx]}\
                        if sample_weights is not None else {}
                    bestimator.fit(x_train, y_train, **fit_params)
                rmtree(cachedir)

                self.bootstrap_fits_[estimator].append(bestimator)

        self._collect_bootstrap_metrics(X, Y)
        self._collect_bootstrap_roc_curves(X, Y)
        return self


    def _collect_bootstrap_metrics(self, x, y):

        def make_inner_d():
            return {s:[] for s in [
                'estimator', 'roc_auc', 'kappa', 'acc', 'f1', 'prec', 'rec',
                'spec', 'split'
            ]}

        for estimator in self.bootstrap_fits_.keys():
            assert len(self.bootstrap_fits_[estimator]) == self.bootstrap_test_gen.n_splits
        
        test_scores = None
        for estimator in self.bootstrap_fits_.keys():

            for split_i, (_, test_idx) in enumerate(self.bootstrap_test_gen.split(x, y)):
            
                x_test, y_test = x.iloc[test_idx], y[test_idx]

                d = make_inner_d()
                
                bestimator = self.bootstrap_fits_[estimator][split_i]
                
                # hard coded for binary classification!!
                y_proba   = bestimator.predict_proba(x_test)[:, -1]
                y_predict = bestimator.predict(x_test)
                
                d['estimator'].append(estimator)
                d['split'].append(split_i)
                
                d['roc_auc'].append(roc_auc_score(y_test, y_proba))
                d['acc'].append(balanced_accuracy_score(y_test, y_predict))
                d['f1'].append(f1_score(y_test, y_predict))
                d['prec'].append(precision_score(y_test, y_predict))
                d['rec'].append(recall_score(y_test, y_predict))
                d['spec'].append(specificity(y_test, y_predict))
                d['kappa'].append(cohen_kappa_score(y_test, y_predict))
            
                d = pd.DataFrame(d)
                test_scores = d if test_scores is None else test_scores.append(d)

        self.bootstrap_scores_ = test_scores.set_index(
            ['estimator', 'split']
        ).sort_index()
        self.bootstrap_scores_gb_ = self.bootstrap_scores_.groupby(level=0)[metric_cols]
        
        estimators = self.bootstrap_scores_.reset_index()['estimator'].unique()
        cis = self.bootstrap_scores_gb_.aggregate([
            lcb, np.mean, ucb, wilcoxon_pvalue
        ]).loc[estimators]
        cis.columns = cis.columns.set_names(['Metric', 'Statistic'])
        cis.index.name = 'Estimator'
        cis = cis.sort_values(('roc_auc', 'lcb'), ascending=False)
        self.bootstrap_scores_summary_ = cis


    def _collect_bootstrap_roc_curves(self, x, y):
        rocs = None

        for split_i, (_, test_idx) in enumerate(
            self.bootstrap_test_gen.split(x,y)
        ):
            x_test, y_test = x.iloc[test_idx], y[test_idx]

            for estimator, fits_list in self.bootstrap_fits_.items():
                be = fits_list[split_i]
                y_proba   = be.predict_proba(x_test)[:, -1]

                rocurve = pd.DataFrame(columns=['split', 'estimator', 'fpr', 'tpr'])
                fpr, tpr, thr = roc_curve(y_test, y_proba, drop_intermediate=True)
                rocurve['fpr'] = fpr; rocurve['tpr'] = tpr
                rocurve['split'] = split_i; rocurve['estimator'] = estimator

                rocs = rocurve if rocs is None else rocs.append(rocurve)
        
        self.bootstrap_roc_curves_ = rocs.set_index(['estimator', 'split'])
        return self


    def bootstrap_boxplot_roc_auc(self):
        res = self.bootstrap_scores_.reset_index().pivot(
            index='split', columns='estimator', values='roc_auc'
        )[self.bootstrap_scores_.index.levels[0]]
        res.columns.name = 'Estimator'
        res.rename(columns=rename_dict_estimator, inplace=True)

        ax = sns.boxplot(data=res, orient='h', palette='GnBu', whis=.9)
        # ax = sns.swarmplot(data=res30, orient='h', palette='magma')
        ax.set_xlabel('Test ROC AUC')
        ax.set_title(f'Distribution of ROC AUC results\non {self.bootstrap_test_gen.n_splits} bootstrapped test sets\n')
        return ax
        # ax.figure.savefig(os.path.join(save_dir, 'roc_boxplot.png'), dpi=500, bbox_inches='tight')


    def bootstrap_confidence_interval_table(self, save_dir=None):
        result = self.bootstrap_scores_summary_.rename(
            columns={**rename_dict_metric, **rename_dict_statistic},
            index=rename_dict_estimator
        )
        if save_dir is not None:
            result.to_csv(os.path.join(save_dir, 'bootstrap.ci.csv'))
        return result


    def fit(self, X, Y, save_dir, sample_weights=None, cvgen=None):
        self._register_feature_set(X, save_dir)
        self.fits_ = {}



        mlflow.sklearn.autolog()

        # please reload
        for estimator, pp in tqdm(ec.items()):

            pipeline, param_grid = pp['pipeline'], pp['param_grid']
            pipeline, param_grid = self._prepare_pipeline(pipeline, param_grid)

            cachedir = mkdtemp()
            pipeline = pipeline.set_params(memory=cachedir)

            print('fitting', rename_dict_estimator[estimator])

            try:
                gs = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    cv=self.cvgen if cvgen is None else cvgen,
                    n_jobs=-1,
                    scoring=scorers,
                    refit=self.fit_criterium
                )

                with mlflow.start_run(run_name=f'{estimator}_gs'):
                    if sample_weights is None:  # i.e. no sample weights were meant to be passed
                        self.fits_[estimator] = gs.fit(X, Y)
                    else:

                        # by this logic, classifiers that dont support this are bypassed
                        fit_params = {'estimator__sample_weight': sample_weights}
                        self.fits_[estimator] = gs.fit(X, Y, **fit_params)

            except ValueError as error:
                if 'Invalid parameter class_weight' in str(error):
                    print(f'Skipping fitting estimator {rename_dict_estimator[estimator]} \
                        because parameter class_weight is invalid for it.')
                    continue
                else:
                    print(error)
                    raise ValueError
                
            except TypeError as error:
                if "unexpected keyword argument 'sample_weight'" in str(error):
                    print(f'Skipping fitting estimator {rename_dict_estimator[estimator]} \
                        because argument sample_weight is invalid for it.')
                    continue
                else:
                    print(error)
                    raise TypeError

            finally:
                rmtree(cachedir)

        return self


    def collect_test_set_results(self, test_x, test_y):
        self._collect_test_set_metrics(test_x, test_y)
        self._collect_test_set_roc_curves(test_x, test_y)


    def _collect_test_set_metrics(self, test_x, test_y):
        
        def make_inner_d():
            return {s:[] for s in [
                'Estimator', 'roc_auc', 'kappa', 'acc', 'f1', 'prec', 'rec', 'spec',
            ]}

        d = make_inner_d()
        for estimator, gs in self.fits_.items():

            bestimator = gs.best_estimator_
            y_pred = bestimator.predict(test_x)
            y_proba = bestimator.predict_proba(test_x)[:, -1]

            d['Estimator'].append(estimator);
            d['roc_auc'].append(roc_auc_score(test_y, y_proba))
            d['acc'].append(balanced_accuracy_score(test_y, y_pred))
            d['f1'].append(f1_score(test_y, y_pred))
            d['prec'].append(precision_score(test_y, y_pred))
            d['rec'].append(recall_score(test_y, y_pred))
            d['spec'].append(specificity(test_y, y_pred))
            d['kappa'].append(cohen_kappa_score(test_y, y_pred))

        self.test_set_scores_ = pd.DataFrame(d).set_index('Estimator').sort_values(
            'roc_auc', ascending=False
        )
        return self


    def test_set_scores(self, save_dir=None):
        result = self.test_set_scores_.rename(
            columns=rename_dict_metric,
            index=rename_dict_estimator
        )
        if save_dir is not None:
            result.to_csv(os.path.join(save_dir, 'test.set.scores.csv'))
        return result


    def _collect_test_set_roc_curves(self, test_x, test_y):
        self.test_set_roc_curves_ = {}
        self.test_set_roc_curves_intermediate_ = {}
        for estimator, gs in self.fits_.items():
            y_proba = gs.best_estimator_.predict_proba(test_x)[:, -1]
            fpr, tpr, thr = roc_curve(test_y, y_proba, drop_intermediate=False)
            self.test_set_roc_curves_intermediate_[estimator] = (fpr, tpr, thr)

            fpr, tpr, thr = roc_curve(test_y, y_proba, drop_intermediate=True)
            self.test_set_roc_curves_[estimator] = (fpr, tpr, thr)
        return self


    def plot_roc_curves_test_set(self, plot):
        fig, ax = plt.subplots()
        lw = 3

        for i, est in enumerate(self.test_set_roc_curves_.keys()):
            fpr, tpr, _ = self.test_set_roc_curves_[est]
            ax.plot(
                fpr, tpr,
                lw=lw/(i+1), alpha=.9,
                label=f'{rename_dict_estimator[est]} ({self.test_set_scores_["ROC AUC"][est]:.2f})'
            )

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC AUC on Testing Cohort')
        ax.legend(loc="lower right")
        # ax.savefig(os.path.join(save_dir, 'external_roc_curves.png'), dpi=500, bbox_inches='tight')
        return ax


    def plot_roc_curves_test_set_with_bootstrap(self,
        train_x, train_y,
        write_reduced_legend=True, shade_std=False, save_dir=None
    ):
        from sklearn.metrics import plot_roc_curve, auc
        self.roc_curve_plots_ = {}

        # define colors hier
        color_bootstrap = '#1f77b4'
        color_test = '#ff7f0e'
        color_std = '#9467bd'

        save_dir = os.path.join(save_dir, 'bootstrap_vs_test_rocs') \
            if save_dir is not None else None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        for estimator in self.fits_.keys():

            sns.set_style('darkgrid')
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            fig, ax = plt.subplots(
                # figsize=(8.0, 3.8)
            )

            for i, (train, test) in enumerate(
                self.bootstrap_test_gen.split(train_x, train_y)
            ):
                  
                classifier = self.bootstrap_fits_[estimator][i]
                viz = plot_roc_curve(
                    classifier, train_x.iloc[test], train_y[test],
                    name='ROC fold {}'.format(i),
                    alpha=0.3, lw=1, ax=ax
                )
                # or, alternatively
                # fpr, tpr, thr = roc_curve(train_y[test], classifier.predict_proba(train_x.iloc[test])[:, -1])
                # roc_auc = auc(fpr, tpr)
                # roc_auc curve returns unevenly spaced fpr. Thus, here it is interpolated
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0  # complementary info
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            # ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='k',
            #          alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(
                mean_fpr, mean_tpr, lw=2, color=color_bootstrap, alpha=.8,
                label=f'[Bootstrapped]    Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})'
            )


            # plot 95% confidence interval
            cis = pd.DataFrame(tprs).agg(
                mean_confidence_interval
            ).transpose().rename(
                columns={i:j for i, j in enumerate(['mean', 'lb', 'ub'])}
            )
            cis['lb'].clip(.0, 1.0, inplace=True)
            cis['ub'].clip(.0, 1.0, inplace=True)

            ax.fill_between(
                mean_fpr, cis['lb'], cis['ub'], color=color_bootstrap, alpha=.4,
                label='[Bootstrapped]    95% Confidence Interval'
            )

            
            # plot test ROC
            test_fpr, test_tpr, _ = self.test_set_roc_curves_[estimator]
            ax.plot(
                test_fpr, test_tpr, color=color_test, alpha=.8, lw=2.4,
                label=f'[Testing Cohort]  ROC (AUC = {auc(test_fpr, test_tpr):.2f})'
            )

            # plot +/- 1 std
            if shade_std:
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + 1*std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - 1*std_tpr, 0)
                ax.fill_between(
                    mean_fpr, tprs_lower, tprs_upper, color=color_std, alpha=.2,
                    label='[Bootstrapped]    \u00B1 1 std. dev.'
                )

            ax.set_title(f'{rename_dict_estimator[estimator]} - Receiver Operating Characteristic')


            font_props = {'family': 'monospace', 'size':8}
            if write_reduced_legend:
                handles, labels = ax.get_legend_handles_labels()
                n_splits = self.bootstrap_test_gen.n_splits
                ax.legend(
                    handles[n_splits:], labels[n_splits:],
                    prop=font_props
                )
            else:
                ax.legend(
                    bbox_to_anchor=(1.03, 1), loc='upper left',
                    prop=font_props
                )
            
            if save_dir is not None:
                fig.savefig(
                    os.path.join(save_dir, f'roc_{estimator}.png'),
                    dpi=500, bbox_inches='tight'
                )
            del fig, ax
        return self


    def _old_plot_roc_curves_test_set_with_bootstrap(self, save_dir=None):
        self.roc_curve_plots_ = {}
        save_dir = os.path.join(save_dir, 'bootstrap_rocs') if save_dir is not None else None
        os.makedirs(save_dir, exist_ok=True)
        
        for estimator in self.test_set_roc_curves_.keys():
            plt.figure()
            ax = sns.lineplot(
                data=self.bootstrap_roc_curves_.loc[estimator],
                x='fpr', y='tpr'
            )
            ax.set_title(f'ROC comparison - {rename_dict_estimator[estimator]}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            fpr, tpr = self.test_set_roc_curves_[estimator]
            ax.plot(fpr, tpr)
            bootroc = self.bootstrap_scores_summary_.loc[estimator].loc['roc_auc'].loc['mean']
            ax.legend([
                f'Bootstrapped Test Sets ({bootroc:.2f})',
                f'Testing Cohort ({self.test_set_scores_["ROC AUC"][estimator]:.2f})'
            ])
            self.roc_curve_plots_[estimator] = ax
            
            if save_dir is not None:
                ax.figure.savefig(os.path.join(save_dir, f'{estimator}.png'), dpi=500, bbox_inches='tight')
            del ax
        if save_dir is not None:
            
            print('check the folder', save_dir, 'for the roc curve plots')
        # return self.roc_curve_plots_


    def confusion_matrix_point(
        self,
        estimator_key,
        fpr_x=None,
        normalize=False,
        on_intermediate=True
    ):
        """Computes confusion matrix for a specific False Positive Rate in the ROC curve.
        Specify False Positive Rate in the argument fpr_x.
        ?
        Args:
            estimator_key (str): A key to an estimator
            fpr_x (float, optional): [.0, 1.0] value of the FPR. If not set, it will
            fetch the FPR point where TPR first reaches 1.0. Defaults to None.
            normalize (bool, optional): Whether to return normalized results. Defaults to False.
            on_intermediate (bool, optional): [description]. Defaults to True.

        Returns:
            pandas.DataFrame: confusion matrix
        """
        if on_intermediate:
            test_roc_curves = self.test_set_roc_curves_intermediate_
        else:
            test_roc_curves = self.test_set_roc_curves_

        try:
            fpr, tpr, thr = self.test_set_roc_curves_[estimator_key]
        except KeyError as error:
            print(error, '\n')
            print('Try on of the available keys:', test_roc_curves.keys())
        
        idx = np.where(tpr == 1.)[0][0] \
            if fpr_x is None else np.where(fpr >= fpr_x)[0][0]
        
        print(f"{rename_dict_estimator[estimator_key]}\n")
        print(f'*visual* [tpr, fpr] = [{tpr[idx]:.2f}, {fpr[idx]:.2f}]\n')

        cat_pred = self.fits_[
            estimator_key].best_estimator_.predict_proba(test_x)[:, -1] > thr[idx]

        cm = confusion_matrix(test_y, cat_pred)
        
        tn, fp, fn, tp = cm.ravel()
        cm = pd.DataFrame(
            cm,
            columns=[f'predicted {l}' for l in ('B', 'M')],
            index=[f'true {l}' for l in ('B', 'M')]
        )
        
        print(f'[TPR] :: {tp/(tp+fn):.2f} »« [FPR] :: {1-tp/(tp+fn):.2f}')
        print(f'[TNR] :: {tn/(tn+fp):.2f} »« [FNR] :: {1-tn/(tn+fp):.2f}\n')

        if normalize:
            return cm / (cm.sum().sum())
        else:
            return cm


    def fit_target_shuffling_assessments(
        self, train_x, train_y, test_x=None, test_y=None
    ):
        """[summary]

        Args:
            train_x ([type]): [description]
            train_y ([type]): [description]
            test_x ([type], optional): [description]. Defaults to None.
            test_y ([type], optional): [description]. Defaults to None.
        """
        self.target_shuffling_assessments_ = {}

        if test_x is not None:
            assert test_y is not None
            self.target_shuffling_assessments_bootstrapped_ = {}
        
        for estimator, gs in tqdm(self.fits_.items()):
            tsa = BBTargetShufflingAssessment(
                deepcopy(gs.best_estimator_),
                X=train_x, Y=train_y,
                external_X=test_x, external_Y=test_y,
                # n_trials=1000
            )
            tsa.run_trials()

            if test_x is not None:
                self.target_shuffling_assessments_bootstrapped_[estimator] = tsa
            else:
                self.target_shuffling_assessments_[estimator] = tsa
        
        return self


    def plot_target_shuffling_assessments(self, save_dir=None):
        save_dir = os.path.join(save_dir, 'tsa') if save_dir is not None else None
        os.makedirs(save_dir, exist_ok=True)
        for estimator, tsa in self.target_shuffling_assessments_.items():
            plt.figure()
            ax = tsa.histogram()
            if save_dir is not None:
                save_plot(ax, os.path.join(save_dir, f'tsa_{estimator}.png'))


    def plot_bootstrapped_target_shuffling_assessments(self, save_dir=None):
        save_dir = os.path.join(save_dir, 'tsa') if save_dir is not None else None
        os.makedirs(save_dir, exist_ok=True)
        for estimator, tsa in self.target_shuffling_assessments_bootstrapped_.items():
            plt.figure()
            ax = tsa.histogram()
            if save_dir is not None:
                save_plot(ax, os.path.join(
                    save_dir, f'tsa_bootstrapped_{estimator}.png'
                ))


if __name__ == '__main__':

    x = np.random.normal(size=(100, 10))
    x = pd.DataFrame(x)
    y = np.random.randint(2, size=100)

    from sklearn.preprocessing import StandardScaler
    pprepends = Pipeline([('scaler', StandardScaler())])

    bbcs = BlackBoxClassifierSearch(pipeline_prepends=pprepends)
    bbcs.fit(x, y)
