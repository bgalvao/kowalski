import seaborn as sns
from sklearn.metrics import plot_confusion_matrix as confusion_matrix_plot
from os.path import join
from kowalski.utils.utils import infer_estimator_name

def plot_confusion_matrix(estimator, test_x, test_y, save_dir=None):
    sns.set_style('dark')

    display = confusion_matrix_plot(estimator, test_x, test_y)
    if save_dir is not None:
        display.figure_.savefig(
            join(save_dir, f'cm_{infer_estimator_name(estimator)}.png'),
            dpi=500, bbox_inches='tight'
        )
