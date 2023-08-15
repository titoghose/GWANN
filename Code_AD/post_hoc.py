import os
from scipy.stats import skewnorm
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
from statsmodels.stats.multitest import multipletests

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
        # f'{label}_Chr{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
        f'{label}_Chr{exp_name}_2500bp_summary.csv')
    if greater_is_better:
        summ_df['P'] = summ_df[metric].apply(lambda x: ep.estimate(x)).values
    else:
        summ_df['P'] = summ_df[metric].apply(lambda x: ep.estimate(-1*x)).values

    _, corr_p, _, alpha_bonf = multipletests(summ_df['P'].values, method='bonferroni')
    summ_df[f'P_bonf'] = corr_p
    # summ_df[f'alpha_bonf'] = alpha_bonf
    summ_df[f'P_fdr_bh'] = multipletests(summ_df['P'].values, method='fdr_bh')[1]
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
    summ_df.to_csv(f'./results_{exp_name}/{label}_{metric}_{exp_name}_gene_summary.csv', index=False)
    print(summ_df.sort_values('P').head(10))
    print()

def combine_chrom_summ_stats(chroms:list, label:str, exp_name:str):
    comb_summ_df = []
    for chrom in chroms:
        summ_df = pd.read_csv(
                    f'./NN_Logs/'+
                    f'Chr{chrom}_{label}_Chr{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
                    f'{label}_Chr{exp_name}_2500bp_summary.csv')
        comb_summ_df.append(summ_df)
    comb_summ_df = pd.concat(comb_summ_df)
    comb_summ_path = (f'./NN_Logs/' +
                        # f'{label}_Chr{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
                        f'{label}_Chr{exp_name}_2500bp_summary.csv')
    if not os.path.exists(comb_summ_path):
        comb_summ_df.to_csv(comb_summ_path, index=False)
    else:
        print(f'Summary file exists at: {comb_summ_path}')
        

if __name__ == '__main__':
    label = 'FH_AD'
    exp_name = 'Sens8_00_GS10_v4'
    combine_chrom_summ_stats([7, 9, 11, 13, 15, 17, 19, 21], 
                             label=label, exp_name=exp_name)
    calculate_p_values(label=label, exp_name=exp_name, 
                       metric='Loss', greater_is_better=False)
    calculate_p_values(label=label, exp_name=exp_name, 
                       metric='Acc', greater_is_better=True)
    calculate_p_values(label=label, exp_name=exp_name, 
                       metric='ROC_AUC', greater_is_better=True)