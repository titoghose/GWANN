import datetime
import multiprocessing as mp
import os
import sys
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import tqdm
import yaml
import torch.nn as nn

sys.path.append('/home/upamanyu/GWANN')

import argparse

from GWANN.dataset_utils import PGEN2Pandas, load_data
from GWANN.models import AttentionMask1, GWANNet5
from GWANN.train_model import Experiment
from GWANN.dummy_data import dummy_plink


def create_dummy_pgen(param_folder:str, label:str) -> None:
    """Create dummy pgen files, followed by converting them to CSV files
    and shuffling them multiple times to create new dummy data files.
    Dummy datasets are created for 10, 20, 30, 40, 50 snps with
    different dosage frequencies. The pgen data is created for all
    MATERNAL and PATERNAL ids and then the CSVs are created for each of
    the phenotypes individually. Finally, the SNPs in the created CSV files are
    shuffled multiple times to create multiple (1000) different dummy datasets.

    Parameters
    ----------
    param_folder : str
        Path to parameters folder.
    
    label : str
        Phenotype/label.
    """
    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']
    
    ids = pd.read_csv('params/all_valid_iids.csv', dtype={'iid':str})['iid'].to_list()
    print(len(ids))

    split_data_base = sys_params['DATA_BASE_FOLDER'].split('/')
    split_data_base[-1] = 'Data_Dummy'
    data_base_folder = '/'.join(split_data_base)
    
    phen_cov = pd.read_csv(sys_params['PHEN_COV_PATH'], sep=' ', comment='#',
                           dtype={'ID_1':str})
    phen_cov.rename(columns={'ID_1':'iid'}, inplace=True)
    phen_cov.set_index('iid', inplace=True)

    lock = mp.Manager().Lock()
    
    cnt = 0
    dosage_freqs = [0.02, 0.04, 0.06, 0.08]
    for num_snps in tqdm.tqdm([10, 20, 30, 40, 50], desc='Num_dummy_snps'):
        for dos_freq in dosage_freqs:
            file_prefix = dummy_plink(samples=ids, 
                    num_snps=num_snps, dosage_freq=dos_freq, 
                    out_folder=f'{data_base_folder}/dummy_pgen')
            pg2pd = PGEN2Pandas(prefix=file_prefix)
            pg2pd.psam['IID'] = ids
            pg2pd.psam['FID'] = ids
            
            load_data(pg2pd=pg2pd, phen_cov=phen_cov, gene=f'Dummy{cnt}', 
                    chrom='1', start=0, end=100, buffer=2500, label=label, 
                    sys_params=sys_params, covs=covs, 
                    preprocess=False, lock=lock)
            cnt += 1

    shuffle_dummy_csvs(sys_params['DATA_BASE_FOLDER'], covs)

def shuffle_dummy_csvs(data_folder:str, covs:list) -> None:
    """Shuffle the SNPs in each dummy dataset multiple times to create
    new dummy datasets. Save the dummy datasets into the 'wins'
    subfolder inside the data_folder to get it ready for NN training.

    Parameters
    ----------
    data_folder : str
        Path to dummy data folder.
    covs : list
        List of covariates in the data.
    """
    flist = os.listdir(data_folder)
    flist = [f for f in flist if 'Dummy' in f]
    wins_folder = f'{data_folder}/dummy_wins'
    if not os.path.exists(wins_folder):
        os.mkdir(wins_folder)
    
    np.random.seed(93)
    random_seeds = np.random.randint(0, 10000, size=(1000,))
    i = 0
    for f in tqdm.tqdm(flist, desc='Dummy file', position=0):
        df = pd.read_csv(f'{data_folder}/{f}', index_col=0, comment='#')
        for snum in tqdm.tqdm(range(50), desc='shuffle', position=1, leave=False):
            shuffled_snps = df.iloc[:, :-(len(covs)+1)].values
            np.random.seed(random_seeds[i])
            np.random.shuffle(shuffled_snps)
            df.iloc[:, :-(len(covs)+1)] = shuffled_snps

            data_path_split = f.split('_')
            data_path_split.insert(2, str(snum))
            data_path = f'{wins_folder}/{"_".join(data_path_split)}'
            df.to_csv(data_path)
            i += 1

def model_pipeline(exp_name:str, label:str, param_folder:str, 
                   gpu_list:list, grp_size:int=10) -> None:
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
    
    gene_win_paths = os.listdir(f'{sys_params["DATA_BASE_FOLDER"]}/wins')
    gene_win_paths = [g for g in gene_win_paths if 'Dummy' in g]
    gene_win_df = pd.DataFrame(columns=['chrom', 'gene', 'win', 'win_count'])
    gene_win_df['chrom'] = [p.split('_')[0].replace('chr', '') for p in gene_win_paths]
    gene_win_df['gene'] = [p.split('_')[1] for p in gene_win_paths]
    gene_win_df['win'] = [int(p.split('_')[2]) for p in gene_win_paths]
    gene_win_df['win_count'] = gene_win_df.groupby('gene').transform('count').values
    gene_win_df.sort_values(['gene', 'win', 'win_count'], 
                            ascending=[True, True, False], inplace=True)
    # gene_win_df = gene_win_df.iloc[:5]

    # Setting the model for the Experiment
    model = GWANNet5
    model_params = {
        'grp_size':grp_size,
        'inp':0,
        'enc':8,
        'h':[128, 64],
        'd':[0.3, 0.3],
        'out':16,
        'activation':nn.ReLU,
        'att_model':AttentionMask1, 
        'att_activ':nn.Sigmoid
    }
    hp_dict = {
        'optimiser': 'adam',
        'lr': 1e-4,
        'batch': 256,
        'epochs': 250,
    }
    prefix = label + '_Chr' + exp_name
    exp = Experiment(prefix=prefix, label=label, params_base=param_folder, 
                     buffer=2500, model=model, model_dict=model_params, 
                     hp_dict=hp_dict, gpu_list=gpu_list, only_covs=False,
                     grp_size=grp_size)
    
    # Remove genes that have already completed
    if os.path.exists(exp.summary_f):
        done_genes = pd.read_csv(exp.summary_f) 
        done_genes = done_genes['Gene'].apply(lambda x: x.split('_')[0]).to_list()
        done_genes = set(done_genes)
        print(len(done_genes))
        gene_win_df = gene_win_df.loc[~gene_win_df['gene'].isin(done_genes)]
    print(f'Number of genes left to train: {gene_win_df.shape[0]}')

    genes = {'gene':[], 'chrom':[], 'win':[]}
    genes['gene'] = gene_win_df['gene'].to_list()
    genes['chrom'] = gene_win_df['chrom'].to_list()
    genes['win'] = gene_win_df['win'].to_list()
    print(len(genes['gene']))

    exp.parallel_run(genes=genes)

    e = datetime.datetime.now()
    print('\n\n', (e-s))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    args = parser.parse_args()
    param_folder='/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'

    # Create data 
    # create_dummy_pgen(param_folder=param_folder)

    # Run model training pipeline
    # gpu_list = list(np.repeat([0, 1, 2, 3, 4], 4))
    # for label in ['MATERNAL_MARIONI']:#, 'PATERNAL_MARIONI']:
    #     model_pipeline(label=label, param_folder=param_folder, gpu_list=gpu_list)

    