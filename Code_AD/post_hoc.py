from scipy.stats import skewnorm
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

class EstimatePValue:
    def __init__(self, null_accs:np.ndarray) -> None:
        self.null_accs = null_accs
        self.moments = skewnorm.fit(null_accs)
    
    def plot_null_dist(self, plot_path:str) -> None:
        x = np.linspace(min(self.null_accs), max(self.null_accs), num=1000)
        y = skewnorm.pdf(x, *self.moments)
        plt.hist(self.null_accs, density=True)
        plt.plot(x, y, linewidth=1)
        plt.savefig(plot_path, dpi=100)
        plt.close()

    def estimate(self, acc:float) -> float:
        return skewnorm.sf(acc, *self.moments)

def calculate_p_values(label:str):
    null_df = pd.read_csv(
        f'/home/upamanyu/GWANN/Code_AD/NN_Logs/' + 
        f'{label}_ChrSens1Dummy_GWANNet5_[128,64]_Dr_0.3_LR:0.0001_BS:256_Optim:adam/'+
        f'{label}_ChrSens1Dummy_summary_{label}_2500bp.csv')
    ep = EstimatePValue(null_accs=null_df['Acc'].values)
    ep.plot_null_dist(f'{label}_null_dist.png')

    summ_df = pd.read_csv(
        f'/home/upamanyu/GWANN/Code_AD/NN_Logs/'+
        f'{label}_ChrSens1_GWANNet5_[128,64]_Dr_0.3_LR:0.0001_BS:256_Optim:adam/'+
        f'{label}_ChrSens1_summary_{label}_2500bp.csv')
    summ_df['P'] = summ_df['Acc'].apply(lambda x: ep.estimate(x)).values

    summ_df.to_csv(f'{label}_summary.csv', index=False)
    print(label)
    print('------------')
    print(summ_df.loc[summ_df['P'] < (0.05/20000)])
    print()

if __name__ == '__main__':
    calculate_p_values('MATERNAL_MARIONI')
    calculate_p_values('PATERNAL_MARIONI')