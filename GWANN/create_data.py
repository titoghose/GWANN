import sys
from typing import Optional

import pandas as pd
sys.path.append('.')

import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import shutil
import numpy as np 
import traceback
import multiprocessing as mp
from functools import partial

from GWANN.dataset_utils import PGEN2Pandas, load_data
from GWANN.models import *
from GWANN.train_utils import FastTensorDataLoader

def return_data(gene_dict:dict, chrom:str, lock:mp.Lock, sys_params:dict, 
                covs:list, buffer:int, label:str, only_covs:bool=False, 
                SNP_thresh:int=10000, ret_data:bool=False) -> Optional[tuple]:
    """Invokes the data creation pipeline for a set of genes.

    Parameters
    ----------
    gene_dict : dict
        Dictionaty containing gene name, chromosome, start position and 
        end position. Each value should be a list. 
        Eg. {'names':['G1', 'G2'], 'chrom':['19', '19'], 'start':[100,200],
        'end':[150, 250]}
    chrom : str
        Chromosome name.
    lock : multiprocessing.Lock
        Lock object to aid process synchronisation. 
    sys_params : dict
        Dictionary of system parameters eg. path to data, path to test
        ids etc.
    covs : list
        List of covariates.
    buffer : int
        Number of flanking base pairs to consider as part of the gene 
        while creating the data, by default 2500.
    label : str
        Phenotype label.
    only_covs : bool, optional
        Whether to create data using only covariates (True) or covariates and
        SNPs (False), by default False.
    SNP_thresh : int, optional
        Maximum number of SNPs to consider. If a gene has more than this
        many SNPs, the file will not be created for the gene, by defailt
        10000.
    ret_data : bool, optional
        Whether to return the created data or not, by default False.
    
    Returns
    -------
    tuple or int
        if ret_data == True, tuple of data files 
            (X, y, X_test, y_test, class_weights, data_cols, num_snps)
        if ret_data == False, 0
    """
   
    pgen_prefix = f'{sys_params["RAW_BASE_FOLDER"][chrom]}/UKB_chr{chrom}'
    train_ids = pd.read_csv(sys_params["TRAIN_IDS_PATH"], 
                            dtype={'iid':str})['iid'].to_list()
    test_ids = pd.read_csv(sys_params["TEST_IDS_PATH"],
                           dtype={'iid':str})['iid'].to_list()
    pg2pd = PGEN2Pandas(pgen_prefix, sample_subset=train_ids+test_ids)
    
    phen_cov = pd.read_csv(sys_params['PHEN_COV_PATH'], 
                           sep=' ', dtype={'ID_1':str}, comment='#')
    phen_cov = phen_cov.rename(columns={'ID_1':'iid'})
    phen_cov.index = phen_cov['iid']
    
    for i, gene in enumerate(gene_dict['names']):
        data = None
        try:
            data = load_data(pg2pd, phen_cov, gene, chrom, gene_dict['start'][i], 
                         gene_dict['end'][i], buffer, label, sys_params, covs, 
                         SNP_thresh, only_covs, lock)
        except Exception:
            print(f'[{gene}] - Data creating error. Check {sys_params["DATA_LOGS"]}')
            if lock is not None:
                lock.acquire()
                with open(sys_params['DATA_LOGS'], 'a') as f:
                    f.write(f'\n{gene}\n')
                    f.write('--------------------\n')
                    f.write(traceback.format_exc())
                lock.release()
            else:
                pass

        if data is None:
            continue
        print('Train data shape for {}: {}'.format(gene, data[0].shape))
        
    if ret_data:
        return data
    else:
        return None
    
def create_data_for_run(label:str, chrom:str, glist:Optional[list], 
                        sys_params:dict, covs:list, gene_map_file:str, 
                        buffer:int=2500, SNP_thresh:int=10000, 
                        num_procs_per_chrom:int=2) -> None:
    """Create data files for a set of genes on a chromosome.

    Parameters
    ----------
    label : str
        Phenotyoe label.
    chrom : str, int
        Chromosome.
    glist : Optional[list]
        List of gene symbols to create data for. To create data for all
        genes on the chromosome, pass None.
    sys_params : dict
        Dictionary of system parameters eg. path to data, path to test
        ids etc.
    covs : list
        List of covariates.
    gene_map_file : str
        Path to the file containing the map of genes to their annotations.
    buffer : int, optional
        Number of flanking base pairs to consider as part of the gene 
        while creating the data, by default 2500.
    num_procs_per_chrom : int, optional
        Number of CPU cores to assign for the task, by default 2.
    """

    genes_df = pd.read_csv(gene_map_file)
    genes_df.drop_duplicates(['symbol'], inplace=True)
    genes_df = genes_df.astype({'chrom':str})
    genes_df.set_index('symbol', drop=False, inplace=True)
    genes_df = genes_df.loc[genes_df['chrom'] == str(chrom)]
    if len(glist) is not None:
        genes_df = genes_df.loc[genes_df['symbol'].isin(glist)]
    
    if len(genes_df) == 0:
        print('No genes found with given chrom and glist')
        return
    
    gdict = {'names':[], 'start':[], 'end':[]}
    for _, r in genes_df.iterrows():
        g = r['symbol']
        s = r['start']
        e = r['end']
        data_path = (f'{sys_params["DATA_BASE_FOLDER"]}/'+
                    f'chr{chrom}_{g}_{buffer}bp_{label}.csv')
        if not os.path.isfile(data_path):
            gdict['names'].append(g)
            gdict['start'].append(s)
            gdict['end'].append(e)
        else:
            print(f'Data file for {g} exists at {data_path}')
            continue
        
    if len(gdict['names']) == 0:
        print('No genes left to create data for.')
        return
    
    print(f'Creating data for {len(gdict["names"])} genes')
    
    ds = []
    num_procs = min(len(gdict['names']), num_procs_per_chrom)
    split_idxs = np.array_split(np.arange(len(gdict['names'])), num_procs)
    for sidxs in split_idxs:
        d_win = {}
        for k in gdict.keys():
            d_win[k] = [gdict[k][si] for si in sidxs]
        ds.append(d_win)

    with mp.Pool(num_procs) as pool:
        lock = mp.Manager().Lock()
        par_func = partial(return_data, 
            chrom=chrom, lock=lock, sys_params=sys_params, covs=covs, 
            buffer=buffer, label=label, ret_data=False, SNP_thresh=SNP_thresh)
        pool.map(par_func, ds)
        pool.close()
        pool.join()

def split(genes:list, covs:list, label:str, read_base:str, 
          write_base:str) -> None:
    """Split gene datasets into windows of 50 SNPs.

    Parameters
    ----------
    genes : list
        List of gene data file names.
    covs : list
        List of covariate columns in the data.
    label : str
        Name of the label/phenotype in the data.
    read_base : str
        Base folder to read data from. 
    write_base : str
        Base folder to write data to. 
    """
    
    for gene_file in genes:
        df_path = f'{read_base}/{gene_file}'
        df = pd.read_csv(df_path, index_col=0, comment='#')
        data_cols = df.columns.to_list()
        num_snps = len(data_cols) - len(covs) - 1
        snp_win = 50
        num_win = int(np.ceil(num_snps/snp_win))
        remaining_snps = num_snps
        # sample_win = np.random.choice(np.arange(0, num_win), 1)
        # sample_win = gsplit[gname]
        for win in range(num_win):
            sind = win * snp_win
            eind = sind+remaining_snps if remaining_snps < snp_win else (win+1)*snp_win
            nsnps = eind-sind
            
            split_f = gene_file.split('_')
            split_f[1] = f'{split_f[1]}_{win}'
            f_win = f'{write_base}/{"_".join(split_f)}'

            cols_win = data_cols[sind+1:eind+1] + covs + [label,]
            df_win = df[cols_win].copy()
            
            # if win == sample_win:
            #     df_win.to_csv(f_win, index=False)
            df_win.to_csv(f_win)

            remaining_snps = remaining_snps - nsnps

def gen_cov_encodings():
    global covs, LABEL

    genes = {
        'names': ['TCF7L2_0',],
        'chrom': ['10',],
        'ids': ['384',]
    }
    lock = mp.Manager().Lock()
    X, y, Xt, yt, class_weights, data_cols, num_snps = return_data(
        lock, genes, ret_data=True)
    
    dev = torch.device('cuda:0')
    
    # model_base = '/home/upamanyu/GWASOnSteroids/NN_Logs/Diabetes_Diag_Covs_GroupAttention_[128,64,16]_Dr_0.3_LR:0.0001_BS:256_Optim:adam'
    model_base = '/home/upamanyu/GWASOnSteroids/NN_Logs/Maternal_Diab_Covs_BMI6PCs_GroupAttention_[128,64,16]_Dr_0.3_LR:0.0001_BS:256_Optim:adam'
    model_path = '{}/{}/{}_{}.pt'.format(model_base, 'APOE', 0, 'APOE')
    cov_model = torch.load(model_path, map_location=torch.device('cpu'))
    cov_model.end_model.linears[-1] = Identity()
    cov_model.to(dev)

    covs_ = copy.copy(covs)
    X = X[:, :, num_snps:]
    Xt = Xt[:, :, num_snps:]
    print(X.shape)
    print(Xt.shape)

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    Xt = torch.from_numpy(Xt).float()
    yt = torch.from_numpy(yt).long()
    
    train_dataloader = FastTensorDataLoader(X, y,
        batch_size=8192, shuffle=False)
    test_dataloader = FastTensorDataLoader(Xt, yt, 
        batch_size=8192, shuffle=False)

    with torch.no_grad():
        train_enc = torch.tensor([])
        test_enc = torch.tensor([])
        for bnum, sample in enumerate(train_dataloader):
            cov_model.eval()
            
            X_batch = sample[0].to(dev)
            cov_enc = cov_model(X_batch).cpu()
            train_enc = torch.cat((train_enc, cov_enc))
        
        for bnum, sample in enumerate(test_dataloader):
            cov_model.eval()
            
            X_batch = sample[0].to(dev)
            cov_enc = cov_model(X_batch).cpu()
            test_enc = torch.cat((test_enc, cov_enc))
        
        print(train_enc.shape)
        print(test_enc.shape)
        np.savez('params/cov_encodings_{}.npz'.format(LABEL), 
            train_enc=train_enc.numpy(), test_enc=test_enc.numpy())


if __name__ == '__main__':

    label='MATERNAL_MARIONI'
    param_folder='./Code_AD/params'
    gene_map_file='./GWANN/datatables/gene_annot.csv'
    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        sys_params = yaml.load(f, Loader=yaml.FullLoader)
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']

    # 1. Create data for only covariates to train the covariate model
    create_data_for_run(
        label=label,
        chrom='19',
        glist=['APOE', 'APOC1', 'TOMM40'],
        param_folder=param_folder,
        gene_map_file=gene_map_file,
        num_procs_per_chrom=3
    )

    # 2. Split data files into windows for training
    # First move data files into a new folder and create an empty
    # folder.
    dp = sys_params['DATA_BASE_FOLDER'].split('/')
    dp[-1] = '_Data'
    new_dp = '/'.join(dp)
    print('Moving data to {}.'.format(new_dp))
    print('Creating new folder {} for data split into windows'.format(
        sys_params['DATA_BASE_FOLDER']))
    shutil.move(sys_params['DATA_BASE_FOLDER'], new_dp)
    os.mkdir(sys_params['DATA_BASE_FOLDER'])
    
    # Next, parallely invoke split_data
    glist = os.listdir(new_dp)
    num_procs = min(20, len(glist))
    glist = np.array_split(glist, num_procs)
    lock = mp.Manager().Lock()
    base_folder = '/'.join(dp[:-1])
    with mp.Pool(num_procs) as pool:
        par_split = partial(split, sys_params, covs, label, 
            '/'.join(dp[:-1]), lock)
        pool.map(par_split, glist)
