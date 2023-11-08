import datetime
import multiprocessing as mp
import os
import sys
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
import torch.nn as nn

sys.path.append('/home/upamanyu/GWANN')

import argparse

from GWANN.dataset_utils import PGEN2Pandas, load_data
from GWANN.models import MLP, BranchedMLP
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
    
    ids = pd.read_csv(f'{sys_params["PARAMS_PATH"]}/all_ids_{label}.csv', 
                      dtype={'iid':str})['iid'].to_list()
    print(len(ids))

    split_data_base = sys_params['DATA_BASE_FOLDER'].split('/')
    split_data_base[-1] = 'Data_Dummy'
    data_base_folder = '/'.join(split_data_base)
    
    phen_cov = pd.read_csv(sys_params['PHEN_COV_PATH'], sep=' ', comment='#',
                           dtype={'ID_1':str})
    phen_cov.rename(columns={'ID_1':'iid'}, inplace=True)
    phen_cov.set_index('iid', inplace=True)

    # lock = mp.Manager().Lock()
    
    # cnt = 0
    # dosage_freqs = [0.02, 0.04, 0.06, 0.08]
    # num_snps = np.arange(0, 501, 25)
    # num_snps[0] = 10
    # for num_snps in tqdm.tqdm(num_snps, desc='Num_dummy_snps'):
    #     for dos_freq in dosage_freqs:
    #         file_prefix = dummy_plink(samples=ids, 
    #                 num_snps=num_snps, dosage_freq=dos_freq, 
    #                 out_folder=f'{data_base_folder}/dummy_pgen')
    #         pg2pd = PGEN2Pandas(prefix=file_prefix)
    #         pg2pd.psam['IID'] = ids
    #         pg2pd.psam['FID'] = ids
            
    #         load_data(pg2pd=pg2pd, phen_cov=phen_cov, gene=f'Dummy{cnt}', 
    #                 chrom='1', start=0, end=100, buffer=2500, label=label, 
    #                 sys_params=sys_params, covs=covs, 
    #                 preprocess=False, lock=lock, save_data=True)
    #         cnt += 1

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
    random_seeds = np.random.randint(0, 10000, size=(5000,))
    # random_seeds = random_seeds[1000:]
    i = 0
    par_func = partial(shuffle_snps, data_folder=data_folder, covs=covs, 
                       wins_folder=wins_folder)
    fargs = []
    num_shuffles = 12
    for f in flist:
        fargs.append((f, random_seeds[i*num_shuffles:(i+1)*num_shuffles]))
        i += 1
    with mp.Pool(len(fargs)) as pool:
        pool.starmap(par_func, fargs)
        
def shuffle_snps(f:str, random_seeds:list, data_folder:str, covs:list, 
                 wins_folder:str) -> None:
    df = pd.read_csv(f'{data_folder}/{f}', index_col=0, comment='#')
    i = 0
    for snum in tqdm.tqdm(range(len(random_seeds)), 
                          desc=f'{f.split("_")[1]} - shuffle'):
        shuffled_snps = df.iloc[:, :-(len(covs)+1)].values
        np.random.seed(random_seeds[i])
        np.random.shuffle(shuffled_snps)
        df.iloc[:, :-(len(covs)+1)] = shuffled_snps

        data_path_split = f.split('_')
        data_path_split.insert(2, str(snum+50))
        data_path = f'{wins_folder}/{"_".join(data_path_split)}'
        df.to_csv(data_path)
        i += 1

def model_pipeline(exp_name:str, label:str, param_folder:str, 
                   gpu_list:list) -> None:
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

    gene_win_paths = os.listdir(f'{sys_params["DATA_BASE_FOLDER"]}/wins')
    gene_win_paths = [g for g in gene_win_paths if 'Dummy' in g]
    gene_win_df = pd.DataFrame(columns=['chrom', 'gene', 'win', 'win_count'])
    gene_win_df['chrom'] = [p.split('_')[0].replace('chr', '') for p in gene_win_paths]
    gene_win_df['gene'] = [p.split('_')[1] for p in gene_win_paths]
    gene_win_df['win'] = [int(p.split('_')[2]) for p in gene_win_paths]
    gene_win_df['win_count'] = gene_win_df.groupby('gene').transform('count').values
    gene_win_df.sort_values(['gene', 'win', 'win_count'], 
                            ascending=[True, True, False], inplace=True)
    
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
                    f'{label}_Cov{exp_name.replace("Dummy", "")}_{model_name}/'+
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
    
    # Remove genes that have already completed
    if os.path.exists(exp.summary_f):
        done_genes_df = pd.read_csv(exp.summary_f) 
        print(done_genes_df.shape)
        gene_win_df['gene_win'] = gene_win_df.apply(lambda x:f'{x["gene"]}_{x["win"]}', axis=1).values
        gene_win_df = gene_win_df.loc[~gene_win_df['gene_win'].isin(done_genes_df['Gene'])]
    
    print(f'Number of genes left to train: {gene_win_df.shape[0]}')
    
    gene_win_df.sort_values(['gene', 'win', 'win_count'], 
                            ascending=[True, True, False], inplace=True)
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
    parser.add_argument('--label', type=str, required=True)
    args = parser.parse_args()
    label = args.label
    
    param_folder='/home/upamanyu/GWANN/Code_AD/params/architecture_testing'
    
    # Dummy data creation
    # create_dummy_pgen(param_folder=param_folder,
    #                   label=label)
    
    # Run model training pipeline
    gpu_list = list(np.tile([0, 1, 2, 3, 4], 5))
    grp_size = 10
    torch_seed=int(os.environ['TORCH_SEED'])
    exp_name = f'ArchTest_{torch_seed}_FrozenCov'
    model_pipeline(exp_name=f'Dummy{exp_name}', label=label, 
                param_folder=param_folder, gpu_list=gpu_list)
    
