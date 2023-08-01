import datetime
import multiprocessing as mp
import os
import sys
from functools import partial
from typing import Optional, Union
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import yaml
import torch.nn as nn

sys.path.append('/home/upamanyu/GWANN')

import argparse

from GWANN.dataset_utils import create_data_for_run, split
from GWANN.models import AttentionMask1, GWANNet5
from GWANN.train_model import Experiment


def create_csv_data(label:str, param_folder:str, chrom:str, SNP_thresh:int=10000,
                    glist:Optional[list]=None, num_procs:int=20, 
                    split:bool=True) -> None:
    
    gene_map_file='/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv'
    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']

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
                   gpu_list:list, glist:list=None, grp_size:int=10, 
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
        'covs':0,
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
    exp = Experiment(prefix=prefix, label=label, params_base=param_folder, 
                     buffer=2500, model=model, model_dict=model_params, 
                     hp_dict=hp_dict, gpu_list=gpu_list, only_covs=False,
                     grp_size=grp_size)
    
    if not shap_plots:
        gene_win_paths = os.listdir(f'{sys_params["DATA_BASE_FOLDER"]}/wins')
        gene_win_paths = [gwp for gwp in gene_win_paths if 'Dummy' not in gwp]
        
        if glist is not None:
            paths = []
            for gwp in gene_win_paths:
                if gwp.split('_')[1] in glist:
                    paths.append(gwp)
            gene_win_paths = paths
        
        gene_win_paths = list(set(gene_win_paths))
        gene_win_paths = [gwp for gwp in gene_win_paths if 'Dummy' not in gwp]
        gene_win_df = pd.DataFrame(columns=['chrom', 'gene', 'win', 'win_count'])
        gene_win_df['chrom'] = [p.split('_')[0].replace('chr', '') for p in gene_win_paths]
        gene_win_df['gene'] = [p.split('_')[1] for p in gene_win_paths]
        gene_win_df['win'] = [int(p.split('_')[2]) for p in gene_win_paths]
        gene_win_df['win_count'] = gene_win_df.groupby('gene').transform('count').values
        gene_win_df.sort_values(['gene', 'win', 'win_count'], 
                                ascending=[True, True, False], inplace=True)
        gene_win_df.drop_duplicates(['gene', 'win'], inplace=True)
        
        print(f'Number of gene win data files found: {gene_win_df.shape[0]}')

        # Remove genes that have already completed
        if os.path.exists(exp.summary_f):
            done_genes_df = pd.read_csv(exp.summary_f) 
            print(done_genes_df.shape)
            gene_win_df['gene_win'] = gene_win_df.apply(lambda x:f'{x["gene"]}_{x["win"]}', axis=1).values
            gene_win_df = gene_win_df.loc[~gene_win_df['gene_win'].isin(done_genes_df['Gene'])]
            
        print(f'Number of genes left to train: {gene_win_df.shape[0]}')
        
        genes = {'gene':[], 'chrom':[], 'win':[]}
        genes['gene'] = gene_win_df['gene'].to_list()
        genes['chrom'] = gene_win_df['chrom'].to_list()
        genes['win'] = gene_win_df['win'].to_list()
        
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--chrom', type=str, required=True)
    args = parser.parse_args()
    label = args.label
    chrom = args.chrom
    param_folder='/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'

    # Create data 
    create_csv_data(label=label, param_folder=param_folder, chrom=chrom)

    # Run model training pipeline
    gpu_list = list(np.repeat([0, 1, 2, 3, 4], 4))
    model_pipeline(label=label, param_folder=param_folder, gpu_list=gpu_list)
    
