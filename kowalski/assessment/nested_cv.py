from os.path import join

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import roc_curve, plot_roc_curve, auc
from sklearn.exceptions import NotFittedError

from kowalski.utils.metrics import scorers
from kowalski.utils.statistics import lcb, ucb, delong_test, mean_confidence_interval
from kowalski.utils.statistics import wilcoxon_pvalue
from kowalski.utils.utils import make_subdirs, infer_estimator_name, read_pkl
from kowalski.utils.evaluate import evaluate_on_test


from scipy.stats import wilcoxon

def wilcoxon_less(differences):
    return wilcoxon(differences, alternative='less').pvalue


_save_subdir = ''

class NestedCV():

    def __init__(
        self, 
        search_estimator:BaseSearchCV,  # this type hint does not make an assertion
        outer_cv=StratifiedKFold(n_splits=5)
    ):
        """[summary]

        Args:
            search_estimator (BaseSearchCV): [description]
            outer_cv ([type], optional): [description]. Defaults to StratifiedShuffleSplit(n_splits=10, test_size=.2, random_state=np.random.randint(2**32-1)).
        """
        self._check_search_estimator(search_estimator)
        self.search_estimator = search_estimator

        self.search_estimator_name_ = infer_estimator_name(self.search_estimator)
        self.outer_cv = outer_cv


    def _check_search_estimator(self, search_estimator):

        # assert that you're actually doing **nested** crossvalidation
        assert issubclass(type(search_estimator), BaseSearchCV)

        scoring = search_estimator.scoring
        refitter = search_estimator.refit

        # if there are multiple metrics to evaluate, make sure that the
        # search_estimator has its refit criteria defined
        # otherwise, you'll get an empty table from cross_validate
        # so this manual check is made here. Refitter should be a string here.
        if callable(scoring):
            if type(scoring()) == dict:
                assert refitter in scoring().keys()
        elif type(scoring) == dict:
            assert refitter in scoring.keys()


    def fit(self, x:pd.DataFrame, y):
        
        # inner_cv happens inside
        _outer_results_ = cross_validate(  # outer is here
            self.search_estimator,  # inner happens here, by passing a *SearchCV object
            x, y,
            n_jobs=-1,
            scoring=self.search_estimator.scoring,
            cv=self.outer_cv,
            return_estimator=True
        )

        # for each split, the resulting best search_estimators are here
        self.fits_ = _outer_results_.pop('estimator')
        
        # outer_cv
        self._outer_results_ = pd.DataFrame(_outer_results_).rename(
            columns=lambda col: 'outer_' + col.replace('test_', '')
        )
        self.outer_results_agg_ = self._outer_results_.aggregate([
            lcb, 'mean', ucb, 'std', wilcoxon_pvalue  # TODO rename to wilcoxon_mean_pvalue
        ])
        
        # inner results
        self._inner_results_ = self._get_inner_results()

        # inspired from scikit learn's example
        self.differences_ = self._get_diff_results()
        self.diff_to_inner_mean_agg_ = self.differences_.agg(
            [lcb, 'mean', ucb, 'std', wilcoxon_less]
        )
    
        #

        # TODO check is fitted, and fit the 'master' if not.
        # self.roc_curves_, self.test_roc_curve_ = self.__collect_roc_curves(
        #     x, y,
        #     test_x, test_y,
        #     self.search_estimator, self.fits_
        # )

        # self.delong_z_, self.delong_pval_ = self.__delong_test(
        #     test_x, test_y,
        #     self.search_estimator,
        #     self.fits_
        # )

        return self


    def _get_inner_results(self):
        
        _inner_results_ = None
        
        for i, fitted_scv in enumerate(self.fits_):
            # cross validation results
            cvr = pd.DataFrame(fitted_scv.cv_results_)

            # best results, (mean)
            br = cvr.loc[fitted_scv.best_index_][
                [metric for metric in cvr.columns if 'mean_' in metric]
            ]
            
            # best results
            br.name=i; br = br.to_frame().transpose()

            # append to result
            _inner_results_ = br if _inner_results_ is None \
                                 else _inner_results_.append(br)
        
        _inner_results_.rename(lambda col: 'inner_'+ col, axis=1, inplace=True)
        return _inner_results_


    def _get_diff_results(self):
        outer = self._outer_results_.rename(lambda s: s.replace('outer_', ''), axis=1)
        inner_mean = self._inner_results_.rename(
                lambda colname: colname.replace('inner_mean_', ''
            ).replace('test_', ''), axis=1)

        diff_to_inner_mean = outer - inner_mean

        if diff_to_inner_mean.isna().any().any():
            raise Exception(f'Diff resulted in empty values\n\n{diff_to_inner_mean}')

        diff_to_inner_mean.rename(lambda s: 'diff_'+s, axis=1, inplace=True)
        return diff_to_inner_mean


    def test_outer_fits(self, test_x: pd.DataFrame, test_y):
        # f this, I already collected on test with estimators fit on the entire
        # learning set!
        result = None
        for i, fitted in enumerate(self.fits_):
            a = pd.Series(evaluate_on_test(fitted, test_x, test_y))
            a.name = i; a = a.to_frame().transpose()
            result = a if result is None else result.append(a)
        result.rename(lambda s: 'test_'+s, axis=1, inplace=True)
        self.outer_fits_on_test_ = result
        return result


    # this is breaking the scope, but fine
    def mlflow_feed(self):
        map(lambda k, v: '_')


    @staticmethod
    def concat(ncvts, save_dir=None):
        """[summary]

        Args:
            ncvts (list or dict): [description]
            save_dir ([type], optional): [description]. Defaults to None.
        """
        # for all of the results, compile them in a single table
        # remember _collect_bootstrap_metrics
        # complete pain in the ass
        assert type(ncvts) == list or type(ncvts) == dict
        if type(ncvts) == list:
            dataframes = {ncvt.search_estimator_name_: ncvt.results_ for ncvt in ncvts}
        else:
            dataframes = {k: ncvt.results_ for k, ncvt in ncvts.items()}
        output = pd.concat(dataframes).unstack(-1)
        if save_dir is not None:
            filepath = join(save_dir, 'nested_cv.stats.html')
            output.to_html(filepath)
        return output
        

    # needs massive change.
    # this will be ported to another class.
    # scope of this class should remain limited to its strict purpose...
    def __collect_roc_curves(self, x, y, test_x, test_y, master_estimator, search_estimators):
        roc_curves = []
        self.validation_probas = []
        assert self.outer_cv.n_splits == len(search_estimators)

        for (_, val_idx), search_estimator in zip(self.outer_cv.split(x,y), search_estimators):
            
            x_val = x.iloc[val_idx]
            y_val = y.iloc[val_idx] if type(y) == pd.Series else y[val_idx]

            y_proba = search_estimator.predict_proba(x_val)[:, -1]
            
            roc_curves.append(roc_curve(y_val, y_score=y_proba))
            self.validation_probas.append(y_proba)

        master_roc_curve = None
        try:
            master_estimator.predict_proba(test_x)[:, -1]

        except NotFittedError:
            print('Master estimator not fitted yet. Fitting now, and it should take a while.')
            master_estimator.fit(x, y)
            
        finally:
            master_roc_curve = roc_curve(
                test_y,
                master_estimator.predict_proba(test_x)[:, -1]
            )

        return roc_curves, master_roc_curve


    def __delong_test(self, test_x, test_y, master_estimator, search_estimators):
        validation_probas = []
        for search_estimator in search_estimators:
            validation_probas.append(search_estimator.predict_proba(test_x)[:, -1].reshape(-1, 1))
        validation_probas =  np.concatenate(validation_probas, axis=1)
        self.mean_proba__ = validation_probas.mean(axis=1)
        return delong_test(
            test_y,
            master_estimator.predict_proba(test_x)[:, -1],
            self.mean_proba__
        )


    def confidence_interval_table(self, save_dir=None):
        return self.results_ 


    def plot_roc_curves(self, save_dir=None):

        # set plotting style here
        sns.set_style('darkgrid')
        color_cv = '#1f77b4'
        color_test = '#ff7f0e'
        color_std = '#9467bd'
        font_props = {'family': 'monospace', 'size':8}

        fig, ax = plt.subplots()
        
        # interpolate cross-validation curves and collect
        tprs = []; aucs = []; mean_fpr = np.linspace(0, 1, 100)
        for fpr, tpr, _ in self.roc_curves_:  # roc curves previously collected ;)
            roc_auc = auc(fpr, tpr)
            # roc_auc curve returns unevenly spaced fpr.
            # Thus, here it is interpolated
            interp_tpr = np.interp(mean_fpr, fpr, tpr); interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr, mean_tpr, lw=2, color=color_cv, alpha=.8,
            label=f'[Validation]  Mean ROC (AUC = {mean_auc:.2f} \u00B1 {std_auc:.2f})'
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
            mean_fpr, cis['lb'], cis['ub'], color=color_cv, alpha=.4,
            label='[Validation]  95% Confidence Interval'
        )
       
        # plot test ROC
        test_fpr, test_tpr, _ = self.test_roc_curve_
        ax.plot(
            test_fpr, test_tpr, color=color_test, alpha=.8, lw=2.4,
            label=f'   [Testing]  ROC (AUC = {auc(test_fpr, test_tpr):.2f})'
        )

        # ax.text(
        #     ax.get_xlim()[-1] - .0275, 0.21, f'DeLong p-value = {self.delong_pval_:.6f}',
        #     horizontalalignment='right', verticalalignment='bottom',
        #     bbox=dict(facecolor='white', alpha=0.5),
        #     **font_props
        # )

        ax.set_title(f'{self.search_estimator_name_} - Receiver Operating Characteristic')
        # return fig, ax

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            prop=font_props
        )
        
        if save_dir is not None:
            fig.savefig(
                join(save_dir, f'roc_{self.search_estimator_name_}.png'),
                dpi=500, bbox_inches='tight'
            )
        del fig, ax  # for some odd reason, it doesn't seem like this is garbage collected

    
