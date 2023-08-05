import os
from scipy.stats import skewnorm
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap

class EstimatePValue:
    def __init__(self, null_accs:np.ndarray, greater_is_better:bool) -> None:
        self.null_accs = null_accs
        self.moments = skewnorm.fit(null_accs)
        self.greater_is_better = greater_is_better
    
    def plot_null_dist(self, plot_path:str) -> None:
        
        plt.hist(self.null_accs, density=True, alpha=0.35)
        for n in [100, 200, 500, 800, 1000]:
            np.random.seed(1047)    
            null_acc_sample = np.random.choice(self.null_accs, size=n)
            moments = skewnorm.fit(null_acc_sample)
            x = np.linspace(min(self.null_accs), max(self.null_accs), num=1000)
            y = skewnorm.pdf(x, *moments)
            plt.plot(x, y, linewidth=2, label=str(n))
        
        plt.legend(title='\n'.join(
            wrap('Distribution estimation sample size', 20)))
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()

    def estimate(self, acc:float) -> float:
        return skewnorm.sf(acc, *self.moments)
        # if self.greater_is_better:
        #     return skewnorm.sf(acc, *self.moments)
        # else:
        #     return skewnorm.cdf(acc, *self.moments)

def calculate_p_values(label:str, exp_name:str, metric:str, greater_is_better:bool):
    if not os.path.exists(f'./results_{exp_name}'):
        os.mkdir(f'./results_{exp_name}')
    
    null_df = pd.read_csv(
        f'./NN_Logs/' + 
        f'{label}_ChrDummy{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
        f'{label}_ChrDummy{exp_name}_2500bp_summary.csv')
    if greater_is_better:
        ep = EstimatePValue(null_accs=null_df[metric].values, 
                            greater_is_better=greater_is_better)
    else:
        ep = EstimatePValue(null_accs=-1*null_df[metric].values, 
                            greater_is_better=greater_is_better)
    ep.plot_null_dist(f'./results_{exp_name}/{label}_{metric}_null_dist.png')

    summ_df = pd.read_csv(
        f'./NN_Logs/'+
        f'{label}_Chr{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
        f'{label}_Chr{exp_name}_2500bp_summary.csv')
    if greater_is_better:
        summ_df['P'] = summ_df[metric].apply(lambda x: ep.estimate(x)).values
    else:
        summ_df['P'] = summ_df[metric].apply(lambda x: ep.estimate(-1*x)).values

    summ_df.to_csv(f'./results_{exp_name}/{label}_{metric}_{exp_name}_summary.csv', index=False)
    
    # summ_df['A1'] = 'A1'
    # summ_df['A2'] = 'A2'
    # summ_df = summ_df[['Gene', 'A1', 'A2', metric, 'P']]
    # summ_df.to_csv(f'./results_{exp_name}/{label}_{exp_name}_METAL_wins_inp.csv', sep=' ', index=False)

    # summ_df['Gene'] = summ_df['Gene'].apply(lambda x:x.split('_')[0]).values
    # summ_df.sort_values(['Gene', 'P'], inplace=True)
    # summ_df.drop_duplicates(['Gene'], inplace=True)
    # summ_df.to_csv(f'./results_{exp_name}/{label}_{exp_name}_METAL_genes_inp.csv', sep=' ', index=False)

    print(label)
    print('------------')
    summ_df['Gene'] = summ_df['Gene'].apply(lambda x:x.split('_')[0]).values
    summ_df.sort_values(['Gene', 'P'], inplace=True)
    summ_df.drop_duplicates(['Gene'], inplace=True)
    print(summ_df.sort_values('P').head(10))
    print()

if __name__ == '__main__':
    calculate_p_values('FH_AD', 'Sens7_00_GS20_v4', 
                       metric='Loss', greater_is_better=False)
    calculate_p_values('FH_AD', 'Sens7_00_GS20_v4', 
                       metric='Acc', greater_is_better=True)
    # calculate_p_values('PATERNAL_MARIONI')