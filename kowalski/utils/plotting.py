import seaborn as sns
from sklearn.metrics import plot_confusion_matrix as confusion_matrix_plot
from os.path import join
from kowalski.utils.utils import infer_estimator_name
import matplotlib.pyplot as plt

# for future modularity, commenting the types of plotting functions here
# ----------------------------------------- estimator-based plotting
def plot_confusion_matrix(estimator, test_x, test_y, save_dir=None):
    sns.set_style('dark')

    display = confusion_matrix_plot(estimator, test_x, test_y)
    if save_dir is not None:
        display.figure_.savefig(
            join(save_dir, f'cm_{infer_estimator_name(estimator)}.png'),
            dpi=500, bbox_inches='tight'
        )


# ----------------------------------------- tabular-based plotting
def prepare_axes(**kwargs):
    
    if 'figsize' in kwargs.keys():
        fig, ax = plt.subplots(figsize=kwargs['figsize'])
        del kwargs['figsize']
    else:
        fig, ax = plt.subplots()

    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
        del kwargs['title']

    return fig, ax, kwargs

def sns_barplot(**kwargs):
    sns.set_theme(style='whitegrid')
    fig, ax, kwargs = prepare_axes(**kwargs)
    defaults = {'ci':95, 'palette':'pastel', 'alpha':.9}
    kwargs = {**defaults, **kwargs}
    return sns.barplot(ax=ax, **kwargs)


def sns_boxplot(**kwargs):
    sns.set_theme(style='whitegrid')
    fig, ax, kwargs = prepare_axes(**kwargs)
    defaults = {'palette':'pastel'}
    kwargs = {**defaults, **kwargs}
    return sns.boxplot(ax=ax, **kwargs)


def sns_stripplot(**kwargs):
    sns.set_theme(style='whitegrid')
    fig, ax, kwargs = prepare_axes(**kwargs)
    if 'ci' in kwargs:
        del kwargs['ci']
    # Show each observation with a scatterplot
    defaults = {'dodge':True, 'alpha':.25, 'zorder':1}
    kwargs = {**defaults, **kwargs}
    return sns.stripplot(**kwargs)
