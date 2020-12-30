
from scipy.stats import entropy


def stacked_bar_plot(df, x:str, y:str='response', normalize:bool=True, indexing=None):
    """
    For readily categorical data
    """
    pvt = df.pivot_table(index=x, columns=y, aggfunc='size').fillna(0)
    if indexing is not None:
        pvt = pvt.loc[indexing]
    pvt_normed = pvt.div(pvt.sum(axis=1), axis=0)

    if normalize:
        ax = pvt_normed.plot.bar(stacked=True, rot=0, colormap='Set1')
    else:
        ax = pvt.plot.bar(stacked=True, rot=0, colormap='Set1')
    
    pvt_normed['entropy'] = pvt_normed.apply(entropy, axis=1, base=2)
    pvt_normed = pvt_normed.round(2)
    pvt_normed['support'] = pvt.sum(1)

    return ax, pvt, pvt_normed



def stacked_histogram(df, x:str, y:str='response'):
    """
    For continuous x
    """
    pass


"""
I am divided between wanting to hack
and wanting to be a good boi researcher
"""

