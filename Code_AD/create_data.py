import sys
sys.path.append('.')

import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import shutil
import numpy as np 
import multiprocessing as mp
from functools import partial

from GWANN.dataset_utils import *
from GWANN.models import *
from GWANN.train_utils import FastTensorDataLoader

def return_data(lock, sys_params, covs, flanking, label, 
        gene_dict, only_covs=False, SNP_thresh=10000, ret_data=False):
    """Invokes the data creation pipeline for a gene.

    Parameters
    ----------
    lock : multiprocessing.Lock
        Lock object to aid process synchronisation. 
    sys_params : dict
        Dictionary of system parameters eg. path to data, path to test
        ids etc.
    covs : list
        List of covariates.
    flanking : int
        Number of flanking base pairs to consider as part of the gene 
        while creating the data, by default 2500.
    label : str
        Phenotype label.
    gene_dict : dict
        Dictionaty containing gene name ('names'), chromosome ('chrom')
        and entrez_id ('id'). Each value should be a singleton list. 
        Eg. {'names':['APOE',], 'chrom':['19',], 'id':[348,]}
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
   
    chrom = gene_dict['chrom'][0]
    geno = None
    phen_cov = [] 
    if int(chrom) <= 10:
        gf = '{}/ukb_imp_chr{}_v3_cleaned_geno_hwe_maf_2.bed'.format(
            sys_params['RAW_BASE_FOLDER'][0], chrom)
        pf = '{}/Variables_UKB.txt'.format(sys_params['RAW_BASE_FOLDER'][0])
    else:
        gf = '{}/ukb_imp_chr{}_v3_cleaned_geno_hwe_maf_2.bed'.format(
            sys_params['RAW_BASE_FOLDER'][1], chrom)
        pf = '{}/Variables_UKB.txt'.format(sys_params['RAW_BASE_FOLDER'][0])
    
    phen_cov = pd.read_csv(pf, sep=' ', dtype={'ID_1':np.int})
    phen_cov = phen_cov.rename(columns={'ID_1':'iid'})
    phen_cov.index = phen_cov['iid']
    
    geno = read_plink1_bin(gf, ref='a0')
    geno = remove_indel_multiallelic(geno)
    
    for i, gene in enumerate(gene_dict['names']):
        chrom = gene_dict['chrom'][i]
        data = load_data(
            {chrom:geno}, phen_cov, [gene,], [chrom,], label, flanking, 
            sys_params['RUNS_BASE_FOLDER'], 
            sys_params, covs,
            SNP_thresh=SNP_thresh,
            only_covs=only_covs,
            lock=lock)
        if data is None:
            continue
        print('Train data shape for {}: {}'.format(gene, data[0].shape))
        
    if ret_data:
        return data
    else:
        return 0
    
def create_data_for_run(label, chrom, glist, sys_params, covs, gene_map_file, 
    flanking=2500, num_procs_per_chrom=2):
    """Create data files for a set of genes on a chromosome.

    Parameters
    ----------
    label : str
        Phenotyoe label.
    chrom : str, int
        Chromosome.
    glist : list
        List of gene symbols to create data for. To create data for all
        genes on the chromosome, pass an empty list [].
    sys_params : dict
        Dictionary of system parameters eg. path to data, path to test
        ids etc.
    covs : list
        List of covariates.
    gene_map_file : str
        Path to the file containing the map of genes to their annotations.
    flanking : int, optional
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
    if len(glist) != 0:
        genes_df = genes_df.loc[genes_df['symbol'].isin(glist)]

    if len(genes_df) == 0:
        print('No genes found with given chrom and glist')
        return
    
    genes = {
        'names': genes_df['symbol'].tolist()[0:],
        'chrom': genes_df['chrom'].tolist()[0:],
        'ids': genes_df['id'].tolist()[0:]}

    g, c, i = [], [], []
    done = []
    for gi, gene in enumerate(genes['names']):
        chrom = genes['chrom'][gi]
        data_path = '{}/chr{}_{}_{}bp_{}.csv'.format(
            sys_params['DATA_BASE_FOLDER'], chrom, gene, flanking, label)        
        if not os.path.isfile(data_path):
            g.append(gene)
            c.append(chrom)
            i.append(genes['ids'][gi]) 
        else:
            print('Data file for {} exists at {}'.format(gene, data_path))
            done.append(gene)
            continue

    genes = {'names': g, 'chrom': c, 'ids': i}
    if len(genes['names']) == 0:
        print('No genes left to create data for.')
        return
    print('Creating data for {} genes'.format(len(genes['names'])))
    assert len(np.unique(genes['chrom'])) == 1, print('More than one chromosome found, aborting!')

    i = 0
    ds = []
    num_procs = min(len(genes['names']), num_procs_per_chrom)
    gene_wins = np.array_split(genes['names'], num_procs)
    chrom_wins = np.array_split(genes['chrom'], num_procs)
    ind_wins = np.array_split(genes['ids'], num_procs)
    for gw, cw, iw in zip(gene_wins, chrom_wins, ind_wins):
        d_win = {}
        d_win['names'] = gw
        d_win['chrom'] = cw
        d_win['ids'] = iw
        ds.append(d_win)

    with mp.Pool(num_procs) as pool:
        lock = mp.Manager().Lock()
        par_func = partial(return_data, 
            lock, sys_params, covs, flanking, label,
            ret_data=False, SNP_thresh=10000)
        _ = pool.map(par_func, ds)
        pool.close()
        pool.join()

def split(sys_params, covs, label, base_folder, lock, gs):
    
    for g in gs:
        gname = g.split('_')[1]
        df_path = '{}/_Data/{}'.format(base_folder, g)
        df = pd.read_csv(df_path)
        
        data_cols = df.columns.to_list()

        temp_df = df.copy()
        temp_df.dropna(
            axis='columns', thresh=np.ceil(0.95*len(temp_df)), inplace=True)
        assert len(temp_df) <= len(df)

        retained_cols = temp_df.columns.to_list()
        retained_cols = set(retained_cols).union(set(covs))
        retained_cols = list(retained_cols.union(set(['iid', label])))
        
        cols = [d for d in data_cols if d in retained_cols]
        df = df[cols]

        num_snps = len([c for c in cols if 'rs' in c])
        snp_win = 50
        num_win = int(np.ceil(num_snps/snp_win))
        remaining_snps = num_snps
        # sample_win = np.random.choice(np.arange(0, num_win), 1)
        # sample_win = gsplit[gname]

        if lock is not None:
            lock.acquire()
            nsdf = pd.read_csv(
                '{}/num_snps.csv'.format(sys_params['RUNS_BASE_FOLDER']))
            nsdf.loc[nsdf['Gene'] == gname, 'num_snps'] = num_snps
            nsdf.to_csv(
                '{}/num_snps.csv'.format(sys_params['RUNS_BASE_FOLDER']), 
                index=False)
            lock.release()

        f_split = g.split('_')
        for win in range(num_win):
            sind = win * snp_win
            eind = sind+remaining_snps if remaining_snps < snp_win else (win+1)*snp_win
            nsnps = eind-sind
            
            f_split_win = copy.copy(f_split)
            f_split_win[1] = '{}_{}'.format(f_split_win[1], win)
            f_win = '{}/Data/{}'.format(base_folder, '_'.join(f_split_win))

            cols_win = ['iid',] + cols[sind+1:eind+1] + covs + [label]
            df_win = df[cols_win].copy()
            assert nsnps == len([c for c in df_win.columns.to_list() if 'rs' in c])
            
            # if win == sample_win:
            #     df_win.to_csv(f_win, index=False)
            df_win.to_csv(f_win, index=False)

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
