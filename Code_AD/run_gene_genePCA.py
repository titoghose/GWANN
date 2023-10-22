import sys
sys.path.append('/home/upamanyu/GWANN')

from GWANN.models import AttentionMask1, GWANNet5
from GWANN.train_model import Experiment

import argparse
import datetime
import os

import numpy as np
import pandas as pd
import yaml
import torch.nn as nn

def model_pipeline(exp_name:str, label:str, param_folder:str, 
                   gpu_list:list, region_dict:dict, grp_size:int=10, 
                   shap_plots:bool=False) -> None:
    """Invoke model training pipeline.

    Parameters
    ----------
    exp_name : str
        Name of the experiment.
    label : str
        Label/phenotype.
    param_folder : str
        Path to params folder.
    gpu_list : list
        List of available gpus to use.
    """
    s = datetime.datetime.now()

    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    
    # Setting the model for the Experiment
    model = GWANNet5
    model_params = {
        'grp_size':grp_size,
        'snps':0,
        'cov_model':None,
        'enc':8,
        'h':[32, 16],
        'd':[0.5, 0.5],
        'out':8,
        'activation':nn.ReLU,
        'att_model':AttentionMask1
    }
    hp_dict = {
        'optimiser': 'adam',
        'lr': 5e-3,
        'batch': 256,
        'epochs': 250,
        'early_stopping':20
    }

    prefix = label + '_Chr' + exp_name

    cov_model_id = f'{prefix.replace("Chr", "Cov")}_GroupAttention_[32,16,8]_Dr_0.5_LR:0.0001_BS:256_Optim:adam/22:16488635-18622343/0_22:16488635-18622343.pt'
    cov_model_path = '{}/{}'.format(sys_params["LOGS_BASE_FOLDER"], cov_model_id)

    exp = Experiment(prefix=prefix, label=label, params_base=param_folder, 
                     buffer=2500, model=model, model_dict=model_params, 
                     hp_dict=hp_dict, gpu_list=gpu_list, only_covs=False,
                     cov_model_path=cov_model_path, grp_size=grp_size)
    
    if not shap_plots:
        gdf = pd.read_csv('../GWANN/datatables/gene_annot.csv', dtype={'chrom':str})
        gdf.set_index('symbol', inplace=True)
        
        region_df = pd.DataFrame.from_dict(region_dict)
        region_df.sort_values(['chrom', 'start'], inplace=True)
        
        print(f'Number of genomic regions found: {region_df.shape[0]}')

        # Remove genes that have already completed
        if os.path.exists(exp.summary_f):
            done_regions_df = pd.read_csv(exp.summary_f) 
            print(f'Number of genomic regions completed: {done_regions_df.shape[0]}')
            
            region_df = region_df.loc[~region_df['gene'].isin(done_regions_df['Gene'])]

        regions = region_df.to_dict(orient='list')
        
        print(f'Number of regions left to train: {len(regions["gene"])}')

        exp.parallel_run(genes=regions)

    # if shap_plots:
    #     os.makedirs(f'results_{exp_name}/shap', exist_ok=True)
    #     for gdict in glist:
    #         fig_name = f'results_{exp_name}/shap/{label}_{gdict["gene"]}_shap.png'
    #         if os.path.isfile(fig_name):
    #             continue
    #         shap_fig = exp.calculate_shap(gene_dict=gdict, device=gpu_list[0])
    #         shap_fig.savefig(fig_name, dpi=100)
    #         plt.close()
        

    e = datetime.datetime.now()
    print('\n\n', (e-s))

def get_chrom_intervals(chrom:str, param_folder:str) -> dict:
    
    with open(f'{param_folder}/params_FH_AD.yaml', 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    
    pca_metadata = pd.read_csv(f'{sys_params["PCA_BASE_FOLDER"]}/{chrom}/metadata.txt', 
                               sep='\t', header=None)
    pca_metadata.columns = ['pca_path', 'evr']
    
    pca_metadata['chrom'] = chrom
    pca_metadata['start'] = pca_metadata['pca_path'].apply(lambda x:int(x.split('_')[2])).values
    pca_metadata['end'] = pca_metadata['pca_path'].apply(lambda x:int(x.split('_')[3].split('.')[0])).values
    pca_metadata['gene'] = pca_metadata.apply(lambda x:f'{x["chrom"]}:{x["start"]}-{x["end"]}', axis=1).values

    return pca_metadata[['gene', 'chrom', 'start', 'end']].to_dict(orient='list')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--chrom', type=str, required=True)
    args = parser.parse_args()
    label = args.label
    chrom = args.chrom
    
    param_folder='/home/upamanyu/GWANN/Code_AD/params/rerun_GenePCA'
    gpu_list = list(np.tile([0, 1, 2, 3, 4], 1))
    grp_size = 10
    torch_seed=int(os.environ['TORCH_SEED'])
    random_seed=int(os.environ['GROUP_SEED'])
    exp_name = f'GenePCA_{torch_seed}{random_seed}_GS{grp_size}_v1'
    regions = get_chrom_intervals(chrom, param_folder)
    model_pipeline(exp_name=exp_name, label=label, 
                   param_folder=param_folder, 
                   gpu_list=gpu_list, region_dict=regions, 
                   grp_size=grp_size, shap_plots=False)
    
