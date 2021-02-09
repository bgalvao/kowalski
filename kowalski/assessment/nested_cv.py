from os.path import join

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, plot_roc_curve, auc

from kowalski.utils.metrics import scorers
from kowalski.utils.statistics import lcb, ucb, delong_test, mean_confidence_interval
from kowalski.utils.statistics import wilcoxon_pvalue as wilcoxon
from kowalski.utils.utils import make_subdirs, infer_estimator_name, read_pkl


_save_subdir = ''

class NestedCVTestAssessment():

    def __init__(
        self, 
        estimator,
        cvgen=StratifiedShuffleSplit(
            n_splits=10, test_size=.2,
            random_state=np.random.randint(2**32-1)
        )
    ):
        self.estimator = estimator
        self.estimator_name_ = infer_estimator_name(estimator)
        self.cvgen = cvgen


    def fit(self, x:pd.DataFrame, y, test_x: pd.DataFrame, test_y):
        cv_results_ = cross_validate(
            self.estimator, x, y,
            n_jobs=-1,
            scoring=scorers,
            cv=self.cvgen,
            return_estimator=True
        )
        self.fits_ = cv_results_.pop('estimator')
        
        self.cv_results_ = pd.DataFrame(cv_results_).rename(
            columns=lambda col: col.replace('test_', '')
        )
        self.results_ = self.cv_results_.aggregate([lcb, 'mean', ucb, wilcoxon])
        # self.results_['estimator'] = 
        
        self.roc_curves_, self.test_roc_curve_ = self.__collect_roc_curves(
            x, y,
            test_x, test_y,
            self.estimator, self.fits_
        )

        # self.delong_z_, self.delong_pval_ = self.__delong_test(
        #     test_x, test_y,
        #     self.estimator,
        #     self.fits_
        # )

        return self



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
            dataframes = {ncvt.estimator_name_: ncvt.results_ for ncvt in ncvts}
        else:
            dataframes = {k: ncvt.results_ for k, ncvt in ncvts.items()}
        output = pd.concat(dataframes).unstack(-1)
        if save_dir is not None:
            filepath = join(save_dir, 'nested_cv.stats.html')
            output.to_html(filepath)
        return output
        


    def __collect_roc_curves(self, x, y, test_x, test_y, master_estimator, estimators):
        roc_curves = []
        self.validation_probas = []
        assert self.cvgen.n_splits == len(estimators)

        for (_, val_idx), estimator in zip(self.cvgen.split(x,y), estimators):
            
            x_val = x.iloc[val_idx]
            y_val = y.iloc[val_idx] if type(y) == pd.Series else y[val_idx]

            y_proba = estimator.predict_proba(x_val)[:, -1]
            
            roc_curves.append(roc_curve(y_val, y_score=y_proba))
            self.validation_probas.append(y_proba)

        master_roc_curve = roc_curve(
            test_y,
            master_estimator.predict_proba(test_x)[:, -1]
        )     
        return roc_curves, master_roc_curve


    def __delong_test(self, test_x, test_y, master_estimator, estimators):
        validation_probas = []
        for estimator in estimators:
            validation_probas.append(estimator.predict_proba(test_x)[:, -1].reshape(-1, 1))
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

        ax.set_title(f'{self.estimator_name_} - Receiver Operating Characteristic')
        # return fig, ax

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles, labels,
            prop=font_props
        )
        
        if save_dir is not None:
            fig.savefig(
                join(save_dir, f'roc_{self.estimator_name_}.png'),
                dpi=500, bbox_inches='tight'
            )
        del fig, ax  # for some odd reason, it doesn't seem like this is garbage collected

    
