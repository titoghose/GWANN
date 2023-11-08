import datetime
import multiprocessing as mp
import os
import sys
from functools import partial
from typing import Optional, Union
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import torch
import yaml
import torch.nn as nn

sys.path.append('/home/upamanyu/GWANN')

import argparse

from GWANN.dataset_utils import create_data_for_run, split
from GWANN.models import MLP, BranchedMLP
from GWANN.train_model import Experiment


def create_csv_data(label:str, param_folder:str, chrom:str, SNP_thresh:int=10000,
                    glist:Optional[list]=None, num_procs:int=20, 
                    split:bool=True) -> None:
    
    gene_map_file='/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv'
    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']

    os.makedirs(sys_params["DATA_BASE_FOLDER"], exist_ok=True)
    create_data_for_run(label, chrom, glist, sys_params, covs, gene_map_file, 
                        buffer=2500, SNP_thresh=SNP_thresh, 
                        num_procs_per_chrom=num_procs, preprocess=False)
    if split:
        create_gene_wins(sys_params=sys_params, covs=covs, label=label, 
                         num_procs=num_procs, genes=glist)

def create_gene_wins(sys_params:dict, covs:list, label:str, 
                     num_procs:int, genes:Optional[list]=None) -> None:
    """Split data files into windows for training. First move data files
    into a new folder and create an empty folder.

    Parameters
    ----------
    sys_params : dict
        _description_
    covs : list
        _description_
    """
    wins_folder = f'{sys_params["DATA_BASE_FOLDER"]}/wins'
    if not os.path.exists(wins_folder):
        print((f'Creating new folder {wins_folder} for data split into windows'))
        os.mkdir(wins_folder)

    glist = [g for g in os.listdir(sys_params["DATA_BASE_FOLDER"]) if g.endswith('.csv')]
    if genes is not None:
        win_files = os.listdir(wins_folder)
        split_glist = [g for g in genes if any([g in wf for wf in win_files])]
        no_split_glist = list(set(genes).difference(set(split_glist)))
        glist = [g for g in glist if any([gene in g for gene in no_split_glist])]
    print(len(glist))
    if len(glist) == 0:
        return
    
    num_procs = min(num_procs, len(glist))
    glist = np.array_split(glist, num_procs)
    with mp.Pool(num_procs) as pool:
        par_split = partial(split, covs=covs, label=label, 
                            read_base=sys_params["DATA_BASE_FOLDER"], 
                            write_base=wins_folder)
        pool.map(par_split, glist)
        pool.close()
        pool.join()

def model_pipeline(exp_name:str, label:str, param_folder:str, 
                   gpu_list:list, glist:Union[list, dict], 
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
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']
    # Setting the model for the Experiment
    # model = MLP
    # model_params = {
    #     'd_in':-1,
    #     'd_out':1,
    #     'd_layers':[256, 256, 256],
    #     'dropout':0.1,
    #     'activation':'ReLU'
    # }
    
    model_name = 'MLP_[32,32,32]_Dr_0.1_LR:0.005_Optim:Adam'
    cov_branch_path = (f'{sys_params["LOGS_BASE_FOLDER"]}/' +
                    f'{label}_Cov{exp_name}_{model_name}/'+
                    f'BCR/0_BCR.pt')
    cov_branch = torch.load(cov_branch_path, map_location='cpu')
    cov_branch.head = None

    model = BranchedMLP
    d_layers = [32, 32, 32]
    model_params = {
        'd_snps':-1,
        'd_layers':d_layers,
        'd_out':1,
        'cov_branch':cov_branch,
        'dropout':0.1,
        'activation':'ReLU'
    }

    hp_dict = {
        'optimiser': 'Adam',
        'lr': 5e-3,
        'train_batch_size': 512,
        'infer_batch_size': 4096,
        'epochs': 250,
        'early_stopping':20,
        'freeze_covs':True
    }

    prefix = label + '_Chr' + exp_name
    exp = Experiment(prefix=prefix, label=label, params_base=param_folder, 
                     buffer=2500, model=model, model_dict=model_params, 
                     hp_dict=hp_dict, gpu_list=gpu_list, only_covs=False)
    
    if not shap_plots:
        gdf = pd.read_csv('../GWANN/datatables/gene_annot.csv', dtype={'chrom':str})
        gdf.set_index('symbol', inplace=True)
        gene_win_dict = {'chrom':[], 'gene':[], 'win':[], 'start':[], 'end':[]}
        if not isinstance(glist, dict):
            gdf = gdf.loc[glist]

            # gdf.sort_index(inplace=True)
            # gdf = gdf.iloc[105:115]
            
            for g, grow in gdf.iterrows():
                wins = list(range(grow['num_wins_2500bp']))
                gene_win_dict['chrom'].extend([grow['chrom']]*len(wins))
                gene_win_dict['gene'].extend([g]*len(wins))
                gene_win_dict['win'].extend(wins)
                gene_win_dict['start'].extend([grow['start']]*len(wins))
                gene_win_dict['end'].extend([grow['end']]*len(wins))

        else:
            gdf = gdf.loc[glist['gene']]
            gene_win_dict['gene'] = glist['gene']
            gene_win_dict['win'] = glist['win']
            print(len(gene_win_dict['gene']), gdf.shape)
            gene_win_dict['chrom'] = gdf['chrom'].values
            gene_win_dict['start'] = gdf['start'].values
            gene_win_dict['end'] = gdf['end'].values
        
        gene_win_df = pd.DataFrame.from_dict(gene_win_dict)
        gene_win_df.sort_values(['gene', 'win'], ascending=[True, True], 
                                inplace=True)
        gene_win_df.drop_duplicates(['gene', 'win'], inplace=True)
        
        print(f'Number of gene win data files found: {gene_win_df.shape[0]}')

        # Remove genes that have already completed
        if os.path.exists(exp.summary_f):
            done_genes_df = pd.read_csv(exp.summary_f) 
            print(f'Number of gene wins completed: {done_genes_df.shape[0]}')
            gene_win_df['gene_win'] = gene_win_df.apply(lambda x:f'{x["gene"]}_{x["win"]}', axis=1).values
            gene_win_df = gene_win_df.loc[~gene_win_df['gene_win'].isin(done_genes_df['Gene'])]

        genes = gene_win_df.to_dict(orient='list')
        
        print(f'Number of genes left to train: {len(genes["gene"])}')

        # X, y, X_test, y_test, cw, data_cols, num_snps = exp.__gen_data__({k:v[-1] for k,v in genes.items()})
        # print(data_cols)
        # exp.__set_genotypes_and_covariates__(chrom=genes['chrom'][0])
        exp.parallel_run(genes=genes)

    if shap_plots:
        os.makedirs(f'results_{exp_name}/shap', exist_ok=True)
        for gdict in glist:
            fig_name = f'results_{exp_name}/shap/{label}_{gdict["gene"]}_shap.png'
            if os.path.isfile(fig_name):
                continue
            shap_fig = exp.calculate_shap(gene_dict=gdict, device=gpu_list[0])
            shap_fig.savefig(fig_name, dpi=100)
            plt.close()
        

    e = datetime.datetime.now()
    print('\n\n', (e-s))

def get_chrom_glist(chrom:str) -> list:
    gdf = pd.read_csv('../GWANN/datatables/gene_annot.csv', dtype={'chrom':str})
    gdf.set_index('symbol', inplace=True)
    gdf = gdf.loc[gdf['chrom'] == chrom]
    gdf = gdf.loc[gdf['num_wins_2500bp'] > 0]
    return gdf.index.to_list()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--chrom', type=str, required=True)
    args = parser.parse_args()
    label = args.label
    chrom = args.chrom
    
    # Run model training pipeline
    param_folder='/home/upamanyu/GWANN/Code_AD/params/architecture_testing'
    gpu_list = list(np.tile([0, 1, 2, 3, 4], 5))
    torch_seed=int(os.environ['TORCH_SEED'])
    exp_name = f'ArchTest_{torch_seed}_FrozenCov'
    # glist = get_chrom_glist(chrom)
    glist = ['APOE', 'BIN1', 'PICALM', 'LRRC7', 'ADAM10', 'APH1B', 'MACROD2', 'WWOX']
    model_pipeline(exp_name=exp_name, label=label, param_folder=param_folder, 
                   gpu_list=gpu_list, glist=glist, shap_plots=False)
    
