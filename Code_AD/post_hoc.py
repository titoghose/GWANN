import os
from scipy.stats import skewnorm
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap

class EstimatePValue:
    def __init__(self, null_accs:np.ndarray) -> None:
        self.null_accs = null_accs
        self.moments = skewnorm.fit(null_accs)
    
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

def calculate_p_values(label:str, exp_name:str):
    if not os.path.exists(f'./results_{exp_name}'):
        os.mkdir(f'./results_{exp_name}')
    
    null_df = pd.read_csv(
        f'./NN_Logs/' + 
        f'{label}_Chr{exp_name}Dummy_GWANNet5_[128,64]_Dr_0.3_LR:0.0001_BS:256_Optim:adam/'+
        f'{label}_Chr{exp_name}Dummy_2500bp_summary.csv')
    ep = EstimatePValue(null_accs=null_df['Acc'].values)
    ep.plot_null_dist(f'./results_{exp_name}/{label}_null_dist.png')

    summ_df = pd.read_csv(
        f'./NN_Logs/'+
        f'{label}_Chr{exp_name}_GWANNet5_[128,64]_Dr_0.3_LR:0.0001_BS:256_Optim:adam/'+
        f'{label}_Chr{exp_name}_2500bp_summary.csv')
    summ_df['P'] = summ_df['Acc'].apply(lambda x: ep.estimate(x)).values

    summ_df.to_csv(f'./results_{exp_name}/{label}_{exp_name}_summary.csv', index=False)
    
    summ_df['A1'] = 'A1'
    summ_df['A2'] = 'A2'
    summ_df = summ_df[['Gene', 'A1', 'A2', 'Acc', 'P']]
    summ_df.to_csv(f'./results_{exp_name}/{label}_{exp_name}_METAL_wins_inp.csv', sep=' ', index=False)

    summ_df['Gene'] = summ_df['Gene'].apply(lambda x:x.split('_')[0]).values
    summ_df.sort_values(['Gene', 'P'], inplace=True)
    summ_df.drop_duplicates(['Gene'], inplace=True)
    summ_df.to_csv(f'./results_{exp_name}/{label}_{exp_name}_METAL_genes_inp.csv', sep=' ', index=False)

    print(label)
    print('------------')
    # print(summ_df.loc[summ_df['P'] < (1e-5)])
    print(summ_df.sort_values('P').head(20))
    print()

if __name__ == '__main__':
    calculate_p_values('MATERNAL_MARIONI', 'Sens6.1')
    # calculate_p_values('PATERNAL_MARIONI')