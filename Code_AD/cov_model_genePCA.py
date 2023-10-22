# coding: utf-8
import os
import sys

sys.path.append('/home/upamanyu/GWANN')

import argparse
import datetime
import multiprocessing as mp
from typing import Optional, Union

import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from GWANN.dataset_utils import load_data, write_and_return_data, load_region_PC_data, PGEN2Pandas
from GWANN.models import AttentionMask1, GroupAttention, Identity
from GWANN.train_model import Experiment
from GWANN.train_utils import FastTensorDataLoader

import torch
import torch.nn as nn

def create_cov_only_data(label:str, param_folder:str) -> None:
    """Create covariate data. By default it will always create the data
    for BCR and RBFOX2 on chromosome 22. Since the data is only
    covariates, it does not matter which gene and chromosome is
    supplied. Two genes are used to ensure later that the NN has
    identical performance. 

    Parameters
    ----------
    label : str
        Label/phenotype.
    param_folder : str
        Path to params folder.
    """
    gene_map_file='/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv'
    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']

    genes_df = pd.read_csv(gene_map_file, comment='#', dtype={'chrom':str})
    genes_df.drop_duplicates('symbol', inplace=True)
    genes_df = genes_df.loc[genes_df['symbol'].isin(['BCR', 'RBFOX2'])]

    lock = mp.Manager().Lock()
    chrom='22'
    write_and_return_data(
        gene_dict=dict(
                    names=genes_df['symbol'].to_list(),
                    chrom=genes_df['symbol'].to_list(),
                    start=genes_df['start'].to_list(),
                    end=genes_df['end'].to_list()),
        chrom=chrom, lock=lock, sys_params=sys_params, covs=covs, buffer=2500, 
        label=label, only_covs=True, SNP_thresh=10_000, ret_data=False, 
        preprocess=True)

def gen_cov_encodings(label:str, param_folder:str, 
                      device:Union[int, str]=0, exp_name:str='') -> None:
    """Generate and save covariate encodings from the penultimate layer 
    of the covariate model. It will always look for a model trained for
    the BCR gene. BCR was selected randomly because the gene does not
    matter for the covariate model.

    Parameters
    ----------
    label : str
        Label/phenotype.
    param_folder : str
        Path to the params folder.
    device : Union[int, str], optional
        PyTorch device, by default 0.
    """
    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']
    
    sys_params['COV_ENC_PATH'] = f'{param_folder}/{exp_name}_cov_encodings_{label}.npz'

    if os.path.isfile(sys_params['COV_ENC_PATH']):
        print(f'Encodings file exists at: {sys_params["COV_ENC_PATH"]}')
        return
    
    model_path=f'{sys_params["LOGS_BASE_FOLDER"]}/{label}_Cov{exp_name}_GroupAttention_[32,16,8]_Dr_0.5_LR:0.0001_BS:256_Optim:adam/BCR/0_BCR.pt'
    cov_model = torch.load(model_path, map_location=torch.device('cpu'))
    cov_model.end_model.linears[-1] = Identity()
    cov_model.to(device)

    X, y, Xt, yt, _, _, _ = load_data(pg2pd=None, phen_cov=None, gene='BCR', 
                    chrom='22', start=0, end=0, buffer=2500, label=label, 
                    sys_params=sys_params, covs=covs, SNP_thresh=10000, 
                    only_covs=True, lock=None)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    Xt = torch.from_numpy(Xt).float()
    yt = torch.from_numpy(yt).long()
    print(f'Training dataset shape: {X.shape}')
    print(f'Testing dataset shape: {Xt.shape}')

    train_dataloader = FastTensorDataLoader(X, y,
        batch_size=8192, shuffle=False)
    test_dataloader = FastTensorDataLoader(Xt, yt, 
        batch_size=8192, shuffle=False)

    with torch.no_grad():
        train_enc = torch.tensor([])
        test_enc = torch.tensor([])
        cov_model.eval()
        for _, sample in tqdm(enumerate(train_dataloader), desc='Train batch'):
            X_batch = sample[0].to(device)
            cov_enc = cov_model(X_batch).cpu()
            train_enc = torch.cat((train_enc, cov_enc))
        
        for _, sample in tqdm(enumerate(test_dataloader), desc='Test batch'):
            X_batch = sample[0].to(device)
            cov_enc = cov_model(X_batch).cpu()
            test_enc = torch.cat((test_enc, cov_enc))
        
        print(f'Train encoding shape: {train_enc.shape}')
        print(f'Test encoding shape: {test_enc.shape}')
        # The order of the saved encodings will be the same as the order
        # of the saved group ids in the param folder
        np.savez(sys_params['COV_ENC_PATH'], 
                 train_enc=train_enc.numpy(), 
                 test_enc=test_enc.numpy())

def model_pipeline(label:str, param_folder:str, gpu_list:list, 
                   exp_name:str='', grp_size:int=10, 
                   shap_plots:bool=False) -> None:
    """Invoke model training pipeline.

    Parameters
    ----------
    label : str
        Label/phenotype.
    param_folder : str
        Path to params folder.
    """
    s = datetime.datetime.now()
    
    prefix = f'{label}_Cov{exp_name}'
    # Setting the model for the Experiment
    model = GroupAttention
    model_dict = {
        'grp_size':grp_size,
        'inp':0,
        'enc':8,
        'h':[32, 16, 8],
        'd':[0.5, 0.5, 0.5],
        'out':2,
        'activation':nn.ReLU,
        'att_model':AttentionMask1
    }
    hp_dict = {
        'optimiser': 'adam',
        'lr': 1e-4,
        'batch': 256,
        'epochs': 250,
        'early_stopping':30
    }
    exp = Experiment(prefix=prefix, label=label, params_base=param_folder, 
            buffer=2500, model=model, model_dict=model_dict, hp_dict=hp_dict, 
            gpu_list=gpu_list, only_covs=True, grp_size=grp_size)

    genes = {'gene':['22:16488635-18622343', '22:18622691-21971296'], 'chrom':['22', '22'], 
             'start':[16488635, 18622691], 'end':[18622343, 21971296]}
    print(genes)

    exp.parallel_run(genes)
    
    if shap_plots:
        gdict = {k:genes[k][0] for k in genes.keys()}
        shap_fig = exp.calculate_shap(gene_dict=gdict, device=gpu_list[0])
        if not os.path.exists(f'results_{exp_name}'):
            os.mkdir(f'results_{exp_name}')
        shap_fig.savefig(f'results_{exp_name}/{label}_cov_model_shap.png', dpi=100)
        plt.close()

    e = datetime.datetime.now()
    print('\n\n', (e-s))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str, required=True)
    args = parser.parse_args()
    label = args.label
    
    param_folder='/home/upamanyu/GWANN/Code_AD/params/rerun_GenePCA'
    gpu_list = [3, 4]
    grp_size = 10
    torch_seed=int(os.environ['TORCH_SEED'])
    random_seed=int(os.environ['GROUP_SEED'])
    exp_name = f'GenePCA_{torch_seed}{random_seed}_GS{grp_size}_v1'
    model_pipeline(label=label, param_folder=param_folder,
                    gpu_list=gpu_list, exp_name=exp_name, 
                    grp_size=grp_size)
    

    # with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
    #     sys_params = yaml.load(f, Loader=yaml.FullLoader)
    
    # with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
    #     covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']

    # pg2pd = PGEN2Pandas('/mnt/sdf/GWANN_pgen/UKB_chr22')
    # load_region_PC_data(pg2pd=pg2pd, phen_cov=None, gene='temp', 
    #                     chrom='22', start=16488635, end=18622343,
    #                     label=label, sys_params=sys_params, covs=covs)