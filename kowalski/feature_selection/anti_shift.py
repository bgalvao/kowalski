import pandas as pd
from tqdm import tqdm
import numpy as np

from scipy.stats import ks_2samp

class AntiDistributionShiftSelector:

    def __init__(self, min_pvalue=.1):

        self.min_pvalue = min_pvalue

    
    def fit(self, train_x:pd.DataFrame, test_x:pd.DataFrame):

        assert train_x.columns == test_x.columns
        dspa = {'ks_statistic':[], 'pvalue':[]}
        for col in train_x.columns:
            result = ks_2samp(train_x[rc], test_x[rc])
            dspa['ks_statistic'].append(result.statistic)
            dspa['pvalue'].append(result.pvalue)

        self.dspa = pd.DataFrame(
            dspa,
            index=train_x.columns
        ).sort_values('pvalue', ascending=False)
        
        self.support = dspa.pvalue >= self.min_pvalue

    
    def transform(self, train_x, test_x):
        assert train_x.columns == test_x.columns
        return 

