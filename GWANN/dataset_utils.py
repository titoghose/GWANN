# coding: utf-8

import os
import csv
import json
import yaml
import copy
import torch
import pickle
import datetime
import itertools
import traceback
import subprocess
import numpy as np
import pandas as pd
from Bio import Entrez
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pandas_plink import read_plink, read_plink1_bin

from GWANN.utils import *

# LD-PCA functions

def get_ld_snps(chrom, snps):
    """Get the list of snps in LD with a set of target snps provided to the
    function. The returned list excludes all target snps. The function
    uses values returned by GCTA-LDF to identify the snps in LD. These
    values need to be precomputed.

    Parameters
    ----------
    chrom : str
        Chromosome the snps are on.
    snps : list
        List of target snps to get the ld-snps list for.

    Returns
    -------
    list
        List of snps in LD with the target snps (excludes all snps from
        the target snps).
    """
    cs_df = pd.read_csv('/mnt/sde/UKBB_LD/LD1/chrom{}.rsq.ld'.format(chrom), 
        sep='\t')
    # print(cs_df.head())
    chrom_snps = cs_df['target_SNP'].to_list()
    
    with open('/mnt/sde/UKBB_LD/LD1/chrom{}.snp.ld'.format(chrom), 'r') as f:
        chrom_ld_list = f.read().splitlines()
    with open('/mnt/sde/UKBB_LD/LD1/chrom{}.r.ld'.format(chrom), 'r') as f:
        chrom_ld_r_list = f.read().splitlines()
    
    # Get all SNPs in LD with the input snps
    snps_ld_ind = [chrom_snps.index(s) for s in snps if s in chrom_snps]
    ld_snps = []
    ld_r = []
    for si in snps_ld_ind:
        csll = chrom_ld_list[si].split(' ')
        csll = [c for c in csll if c not in ['NA', '']]
        
        crll = chrom_ld_r_list[si].split(' ')
        crll = [float(r) for r in crll if r not in ['NA', '']]

        ld_snps.extend(csll)
        ld_r.extend(crll)
    
    # Drop duplicate snps in the list
    ld_snps = np.asarray(ld_snps)
    ld_r = np.asarray(ld_r)
    _, ind = np.unique(ld_snps, return_index=True)
    ld_snps = ld_snps[ind]
    ld_r = ld_r[ind]
    
    # Get SNPs in LD, other than the snps in the input list
    ld_ind = [i for i, s in enumerate(ld_snps) if s not in snps]
    ld_snps = ld_snps[ld_ind]
    ld_r = ld_r[ld_ind]

    # Sort the SNPs based on correlation and retain only 300 most
    # correlated SNPs
    # sorted_ind = np.argsort(ld_r)[-300:]
    # ld_snps = ld_snps[sorted_ind]
    # ld_r = ld_r[sorted_ind]

    ld_snps = [s for s in ld_snps if s.startswith('rs')]

    return ld_snps

def gen_pcs(chrom, snps, gene, train_ids_f, test_ids_f):
    """Create and save LD-PC components and projections for a set of
    snps. Saves 2 files: (i) Projection of principal components for test
    and train sets along with the list of iids matching each row of the
    projection matrix, (ii) Pickle file representing the sklearn PCA
    model used to generate these projections.

    Parameters
    ----------
    chrom : str,int
        Chromosome the snps are on.
    snps : list
        List of snps to calculate the PCA on.
    gene : str
        Gene the snps are in LD with.
    train_ids_f : str
        Path to train ids df
    test_ids_f : str
        Path to test ids df
    """
    train_ids = pd.read_csv(train_ids_f, dtype={'iid':int})['iid'].values
    _, ind = np.unique(train_ids, return_index=True)
    train_ids = np.asarray(train_ids[ind], dtype=str)

    test_ids = pd.read_csv(test_ids_f, dtype={'iid':int})['iid'].values
    _, ind = np.unique(test_ids, return_index=True)
    test_ids = np.asarray(test_ids[ind], dtype=str)
    
    iids = np.asarray(np.concatenate((train_ids, test_ids)), dtype=str)
    
    if int(chrom) <= 10:
        geno = read_plink1_bin('/mnt/sdd/UKBB_1/ukb_imp_chr{}_v3_cleaned_geno_hwe_maf_2.bed'.format(chrom),
            ref='a0')
    else:
        geno = read_plink1_bin('/mnt/sdc/UKBB_2/ukb_imp_chr{}_v3_cleaned_geno_hwe_maf_2.bed'.format(chrom),
            ref='a0')
    
    geno = remove_indel_multiallelic(geno)
    sample = list(np.array(geno['sample']))
    variant = list(np.array(geno['variant']))
    
    # Train and test iid mask to extract only relevant iids from data
    iid_df = pd.DataFrame(np.vstack((sample, np.arange(0, len(sample)))).T, columns=['iid', 'ind'])
    iid_df = iid_df.astype({'iid':str, 'ind':int})
    iid_df.set_index('iid', drop=False, inplace=True)
    train_ind = iid_df.loc[train_ids]['ind'].values
    train_mask = np.zeros(len(iid_df), dtype=bool)
    train_mask[train_ind] = True
    test_ind = iid_df.loc[test_ids]['ind'].values
    test_mask = np.zeros(len(iid_df), dtype=bool)
    test_mask[test_ind] = True
    
    # SNP mask to extract only relevant snps from data
    snp_df = pd.DataFrame(np.vstack((variant, np.arange(0, len(variant)))).T, columns=['snp', 'ind'])
    snp_df = snp_df.astype({'snp':str, 'ind':int})
    snp_df.set_index('snp', drop=False, inplace=True)
    snp_ind = snp_df.loc[snps]['ind'].values
    snp_mask = np.zeros(len(snp_df), dtype=bool)
    snp_mask[snp_ind] = True

    geno_train = geno[train_mask, snp_mask]
    geno_test = geno[test_mask, snp_mask]

    # Convert xarray to numpy array and get iids for each data row
    train_mat = geno_train.values
    train_ids = np.array(geno_train['sample'])
    test_mat = geno_test.values
    test_ids = np.array(geno_test['sample'])
    print(train_mat.shape, test_mat.shape)

    pca = PCA(n_components=10)
    pca.fit(train_mat)
    train_proj = pca.transform(train_mat)
    test_proj = pca.transform(test_mat)
    print(train_proj.shape, test_proj.shape)

    # Save all PC projections with corresponding iids
    np.savez('/mnt/sdb/LD_PCA/{}_chr{}.npz'.format(gene, chrom), 
        train_ids=train_ids, train_pcs=train_proj,
        test_ids=test_ids, test_pcs=test_proj)
    
    # Save PCA model used to generate the PCs
    model_f = '/mnt/sdb/LD_PCA/{}_chr{}_model.pickle'.format(gene, chrom)
    with open(model_f, 'wb') as f:
        pickle.dump(pca, f)

def load_ldpcs(chrom, gene, sys_params, scale=True):
    """Loads and returns saved LD-PCs.

    Parameters
    ----------
    chrom : str, int
        Chromosome the gene is on
    gene : str
        Gene to load the LD-PCs for
    sys_params : dict
        Dictionary containing system specific parameters - necessary
        foder paths.
    scale : bool
        MinMax scale data or not, by default True

    Returns
    -------
    tuple 
        Tuple fo 4 numpy ndarrays corresponding to: 
        0 - train set pcs
        1 - train set ids
        2 - test set pcs
        3 - test set ids
    """
    pcs = np.load('{}/{}_chr{}.npz'.format(
        sys_params['LDPCA_BASE_FOLDER'], gene, chrom),
        allow_pickle=True)
    train_pcs = pcs['train_pcs']
    train_ids = np.array(pcs['train_ids'], dtype=int)
    test_pcs = pcs['test_pcs']
    test_ids = np.array(pcs['test_ids'], dtype=int)

    if scale:
        mms = MinMaxScaler()
        mms.fit(np.concatenate((train_pcs, test_pcs)))
        train_pcs = mms.transform(train_pcs)
        test_pcs = mms.transform(test_pcs)

    return train_pcs, train_ids, test_pcs, test_ids

# Beta matrix generation functions

def get_beta_matrix(snps, chrom, betaf):
    """
    Method still under development
    """
    beta_matrix = torch.zeros((6, len(snps)))
    beta_df = pd.read_csv(betaf)
    beta_df = beta_df.loc[beta_df['SNP'].isin(snps)]
    for s in snps:
        if s not in beta_df['SNP'].values:
            beta_df = beta_df.append({'CHR': chrom, 'SNP':s, 'BETA_mat':float('nan'), 
                'BETA_pat':float('nan'), 'BETA_matpat':float('nan')}, ignore_index=True)
    beta_df.set_index('SNP', drop=False, inplace=True)
    beta_df = beta_df.loc[snps]
    beta_matrix[0, :] = torch.from_numpy(beta_df['BETA_mat'].values)
    beta_matrix[2, :] = torch.from_numpy(beta_df['BETA_pat'].values)
    beta_matrix[4, :] = torch.from_numpy(beta_df['BETA_matpat'].values)
    beta_matrix[1, :] = torch.isnan(beta_matrix[0, :]).float()
    beta_matrix[3, :] = torch.isnan(beta_matrix[2, :]).float()
    beta_matrix[5, :] = torch.isnan(beta_matrix[4, :]).float()
    
    beta_df.fillna(0, inplace=True)
    beta_matrix[0, :] = torch.from_numpy(beta_df['BETA_mat'].values)
    beta_matrix[2, :] = torch.from_numpy(beta_df['BETA_pat'].values)
    beta_matrix[4, :] = torch.from_numpy(beta_df['BETA_matpat'].values)
    beta_matrix = torch.unsqueeze(beta_matrix, 0)
    
    return beta_matrix

# Group Train specific functions

def group_ages(ages, num_grps):
    """Function to convert ages into age groups based on frequency of
    each age. The age groups may not be exactly of the same size,
    but the best possible split will be found. 

    Parameters
    ----------
    ages : ndarray
        Array of ages
    num_grps : int
        Number of equally sized (as far as possible) groups to bin the 
        ages into
    
    Returns
    -------
    ndarray
        Array of binned ages. The new age values are integer
        placeholders representing the age group number, not the actual 
        ages
    """
    u_ages = np.sort(np.unique(ages))
    bins = np.append(u_ages, u_ages[-1]+1)
    n, _ = np.histogram(ages, bins=bins)
    cumsum = np.cumsum(n)
    new_ages = ages.copy()
    prev_grp_end = min(u_ages)-1
    
    for g_num in range(num_grps):
        thresh = round((g_num+1)*len(ages)/num_grps)
        grp_end = list(np.array(cumsum >= thresh, dtype=int)).index(1)
    
        diff1 = cumsum[grp_end] - thresh
        diff2 = thresh - cumsum[grp_end - 1]
        grp_end = grp_end if diff1 < diff2 else grp_end-1
        grp_end = u_ages[grp_end]

        condition = (ages <= grp_end) & (ages > prev_grp_end)
        new_ages = np.where(condition, np.repeat(g_num, len(ages)), new_ages)
        
        print('Group{}: {}-{} yrs'.format(g_num, prev_grp_end, grp_end))
        prev_grp_end = grp_end
        
    
    return new_ages

def group_data_prep(X, y, Xt, yt, grp_size, covs, test_size=0.0, 
        train_oversample=10, test_oversample=10):
    """Convert data arrays to grouped data arrays after balancing as
    best as possible for age and sex.

    Parameters
    ----------
    X : ndarray, None
        Train dataset
    y : ndarray, None
        Train data labels
    Xt : ndarray, None
        Test data set. If None, will create a startified test set from
        the training set based on the test_size parameter
    yt : ndarray, None
        Test data labels. If None, will create a startified test set from
        the training set based on the test_size parameter
    grp_size : int
        Size of groups
    covs : list
        List of covariates
    test_size : float, optional
        Test set split if test data and labels are not provided, 
        by default 0.0
    train_oversample : int, optional
        Factor to oversample all train data samples by before forming into
        groups, by default 10
    test_oversample : int, optional
        Factor to oversample all test data samples by before forming into
        groups, by default 10

    Returns
    -------
    tuple
        Tuple fo 4 ndarrays:
        0 - Grouped train data, None if X or y was None
        1 - Train labels, None if X or y was None
        2 - Grouped test data, None if X (or y) and Xt (or yt) were None
        3 - Test labels, None if X (or y) and Xt (or yt) were None
    """
    age_col = covs.index('f.21003.0.0') - len(covs)
    sex_col = covs.index('f.31.0.0') - len(covs)

    if Xt is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, 
            random_state=625)
        
        for ti, vi in sss.split(X, y):
            Xt = X[vi]
            yt = y[vi]
            X = X[ti]
            y = X[ti]

    Xs = []
    ys = []
    for X_, y_, over in [(X, y, train_oversample), (Xt, yt, test_oversample)]:
        if X_ is None:
            Xs.append(None)
            ys.append(None)
            test_flag = 1
            continue

        case = np.where(y_ == 1)[0]
        cont = np.where(y_ == 0)[0]
        
        # Randomly oversample and interleave the individuals
        case = np.repeat(case, over)
        np.random.seed(1763)
        np.random.shuffle(case)
        cont = np.repeat(cont, over)
        np.random.seed(1763)
        np.random.shuffle(cont)

        # Remove extra samples that will not form a group of the
        # expected size
        case_num = len(case) - len(case)%grp_size
        cont_num = len(cont) - len(cont)%grp_size
        np.random.seed(8983)
        case = np.random.choice(case, case_num, replace=False)
        np.random.seed(8983)
        cont = np.random.choice(cont, cont_num, replace=False)

        # Create groups balanced on age and sex (as close as possible)
        # Xg[0] - Case, Xg[1] - Control
        Xg = [[], []]
        sex = X_[:, sex_col]
        age = X_[:, age_col]
        vprint('ages : {}'.format(np.unique(age)))
        age = group_ages(age, num_grps=3)
        # X_[:, age_col] = age
        for j, label in enumerate([case, cont]):
            n_splits = round(len(label)/grp_size)
            X_label = X_[label]
            sex_label = sex[label]
            age_label = age[label]

            # Combine age groups and sex to form stratified groups
            age_sex_label = np.add(age_label, X_label[:, sex_col]*3)
            
            vprint('age groups: {}'.format(np.unique(age_label)))
            vprint('sex: {}'.format(np.unique(sex_label)))
            vprint('Unique age_sex groups: {}'.format(np.unique(age_sex_label)))
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                random_state=4231)
            for _, ind in skf.split(np.zeros(len(label)), age_sex_label):
                Xg[j].append(X_label[ind])
                # Xg[j].append(X_label[ind][:, [sex_col, age_col]])
                
        Xs.append(np.concatenate((Xg[0], Xg[1])))
        ys.append(np.concatenate((np.ones(len(Xg[0])), np.zeros(len(Xg[1])))))
        test_flag = 1

    
    return Xs[0], ys[0], Xs[1], ys[1]

# Data creation and loading functions

def load_data(genotypes, phen_cov, gene, chrom, label, bp, log_folder, 
        sys_params, covs, over_coeff=0.0, balance=1.0, SNP_thresh=1000,
        only_covs=False, lock=None):
    """Load data, balance it and obtain train-test splits 
    before training.

    Parameters
    ----------
    genotypes : dict
        Dictionary of xarrays, each of which are combined PLINK bed, 
        bim and fam files for a chromosome in the list of chromosomes
        passed. 
    phen_cov : Pandas.DataFrame
        Dataframe containing covariate and phenotype information
    gene : list
        List of genes to include in the dataset. For a single gene 
        pass as a list of a single str
    chrom : list
        List of chromosomes of the genes to include in the dataset. 
        For a single gene pass as a list of a single str.
    label : str
        Prediction label/phenotype for the dataset.
    bp: int
        Base-pair buffer to consider on each side of the genes.
    log_folder : str
        Path to log folder to record dataset errors or logs.
    sys_params : dict
        Dictionary containing system specific parameters - necessary
        foder paths.
    covs : list
       List of covariates
    over_coeff : float, optional
        The percentage to oversample the minority class, by default 0.0
    balance : float, optional
        Desired ratio of control:case in the balanced data, 
        by default 1.0
    SNP_thresh : int, optional
        Maximum number of SNPs. Genes with SNPs greater 
        than this will be dropped, by default 1000
    only_covs : bool, optional
        Return a dataframe of only covariates, without any SNPs.
    lock : mp.Lock, optional
        Lock object to prevent issues with concurrent access during read
        and write, by default None

    Returns
    -------
    tuple
        Data tuple containing the following in the respective
        indices:
        0 - Balanced training data (ndarray)
        1 - Balanced training labels (ndarray)
        2 - Balanced testing data (ndarray)
        3-  Balanced testing label (ndarray)
        4 - Array containing class weights based on training labels
        (ndarray)
        5 - Names of each column in the data arrays (list)
        6 - Number of SNPs in the data arrays (int)
    """
    test_ids_f = sys_params['TEST_IDS_PATH']
    train_ids_f = sys_params['TRAIN_IDS_PATH']
    agg_df = None
    num_snps = 0
    data_cols = []
    for c, g in zip(chrom, gene):
        data_path = '{}/chr{}_{}_{}bp_{}.csv'.format(
            sys_params['DATA_BASE_FOLDER'], c, g, bp, label)
        vprint(data_path)
        # Load data 
        data, snps = dataset_from_raw(genotypes[c], phen_cov, [c,], [g,], 
            None, [bp,], label, None, log_folder, sys_params, covs, bounds=[], 
            SNP_thresh=SNP_thresh, save_file=True, data_fname=data_path)
        
        if data is None:
            continue
        
        g_snps = len(snps)
        num_snps += g_snps
        data_cols.extend(snps)

        if agg_df is None:
            agg_df = data.copy()
        else:
            agg_df = agg_df.merge(data[['iid'] + snps], on='iid', how='inner')
        del(data)
    
    if agg_df is None:
        return None

    mmiss = 0
    tmiss = 0

    # Remove all individuals with missing info for these covariates
    agg_df.dropna(subset=['f.31.0.0', 'f.21003.0.0'], how='any', inplace=True)
    
    # Scale each feature between 0 and 1
    mm_scaler = MinMaxScaler()
    ids = agg_df['iid'].values
    agg_df = pd.DataFrame(mm_scaler.fit_transform(agg_df), 
        columns=agg_df.columns)
    # agg_df.iloc[:, 1:-1] += 1
    agg_df['iid'] = ids
    
    # Creating test set
    test_set = pd.read_csv(test_ids_f)
    test_cases = test_set.loc[test_set[label] == 1]['iid'].values
    agg_df.loc[agg_df['iid'].isin(test_cases), label] = 1
    test_conts = test_set.loc[test_set[label] == 0]['iid'].values
    agg_df.loc[agg_df['iid'].isin(test_conts), label] = 0
    test_ids = np.concatenate((test_cases, test_conts))
    
    # Filtering out test iids from the agg_df
    vprint('Before test set: ', agg_df.shape)
    test_df = agg_df.loc[agg_df['iid'].isin(test_ids)]
    test_miss = np.sum(np.isnan(test_df.iloc[:, 1:1+num_snps].values), axis=0)
        
    agg_df = agg_df.loc[~agg_df['iid'].isin(test_ids)]
    vprint('After test set: ', agg_df.shape)

    
    train_set = pd.read_csv(train_ids_f)
    train_cases = train_set.loc[train_set[label] == 1]['iid'].values
    agg_df.loc[agg_df['iid'].isin(train_cases), label] = 1
    train_conts = train_set.loc[train_set[label] == 0]['iid'].values
    agg_df.loc[agg_df['iid'].isin(train_conts), label] = 0
    train_ids = np.concatenate((train_cases, train_conts))
    agg_df = agg_df.loc[agg_df['iid'].isin(train_ids)]
    train_miss = np.sum(np.isnan(agg_df.iloc[:, 1:1+num_snps].values), axis=0)
    
    # mmiss = np.max((train_miss + test_miss))/(len(agg_df)+len(test_df))
    # tmiss = np.sum((train_miss + test_miss))/((len(agg_df)+len(test_df))*num_snps)
    
    # Dropping snps that are possibly interacting with the covariates
    # pi_snps = pruned_dict[gene[0]]
    # num_snps = len(pi_snps)
    # data_cols = pi_snps

    # Drop all SNPs with more than 5% missing values    
    # print(agg_df.loc[agg_df['f.845.0.0'].isnull() & (agg_df[label] == 1)].shape)
    # print(agg_df.loc[agg_df['f.845.0.0'].isnull() & (agg_df[label] == 0)].shape)
    # print(test_df.loc[test_df['f.845.0.0'].isnull() & (test_df[label] == 1)].shape)
    # print(test_df.loc[test_df['f.845.0.0'].isnull() & (test_df[label] == 0)].shape)
    temp_df = pd.concat((test_df, agg_df))
    # temp_df.dropna(axis='columns', thresh=np.ceil(0.95*len(temp_df)), inplace=True)
    retained_cols = temp_df.columns.to_list()
    retained_cols = list(set(retained_cols).union(set(covs)))
    del(temp_df)

    test_df.fillna(-1, inplace=True)
    agg_df.fillna(-1, inplace=True)
    train_df = agg_df

    data_cols.extend(list(covs))
    data_cols = [d for d in data_cols if d in retained_cols]
    num_snps = len(data_cols) - len(covs)
    # data_cols = ['iid',] + data_cols
    
    # Load and concat LD_PCs to covariates
    # test_df.set_index('iid', inplace=True, drop=False)
    # train_df.set_index('iid', inplace=True, drop=False)
    # trpc, trid, tepc, teid = load_ldpcs(''.join(chrom), ''.join(gene), 
    #     sys_params)
    # assert len(train_df) == len(trpc), print(len(train_df), len(trpc))
    # assert len(test_df) == len(tepc), print(len(test_df), len(tepc))
    # pc_cols = ['LDPC'+str(i) for i in range(1, 6)]
    # train_df.loc[trid, pc_cols] = trpc[:, :len(pc_cols)]
    # test_df.loc[teid, pc_cols] = tepc[:, :len(pc_cols)]
    
    # data_cols.extend(pc_cols)
    if only_covs:
        data_cols = data_cols[num_snps:]
        num_snps = 0

    try:
        grps = np.load('{}/group_ids_{}.npz'.format(
            sys_params['PARAMS_PATH'], label))
        grp_size = grps['train_grps'].shape[1]
        tr_grps = np.asarray(grps['train_grps'].flatten(), dtype=int)
        te_grps = np.asarray(grps['test_grps'].flatten(), dtype=int)
        
        train_df.set_index('iid', inplace=True, drop=False)
        X = train_df.loc[tr_grps][data_cols].to_numpy()
        X = np.reshape(X, (-1, grp_size, len(data_cols)))
        y = grps['train_grp_labels']

        test_df.set_index('iid', inplace=True, drop=False)
        X_test = test_df.loc[te_grps][data_cols].to_numpy()
        X_test = np.reshape(X_test, (-1, grp_size, len(data_cols)))
        y_test = grps['test_grp_labels']
    
    except FileNotFoundError:
        X_test = test_df[data_cols].to_numpy()
        y_test = test_df[label].to_numpy()
        
        X = train_df[data_cols].to_numpy()
        y = train_df[label].to_numpy()

    class_weights = compute_class_weight(class_weight='balanced', 
                                        classes=np.unique(y), y=y)

    if lock is not None:
        lock.acquire()
        try:
            df = pd.read_csv('{}/num_snps.csv'.format(sys_params['RUNS_BASE_FOLDER']))
        except FileNotFoundError:
            df = pd.DataFrame(columns=['Gene', 'chrom', 'bp', 'num_snps', 'label'])
            
        df = df.append({
            'Gene' : ''.join(gene),
            'chrom' : ''.join(chrom),
            'bp' : bp,
            'num_snps' : num_snps,
            'label' : label}, ignore_index=True)
        df.to_csv('{}/num_snps.csv'.format(sys_params['RUNS_BASE_FOLDER']), 
            index=False)
        lock.release()
    
    return X, y, X_test, y_test, class_weights, data_cols, num_snps
    # return mmiss, tmiss

def dataset_from_raw(genotype, phen_cov, chroms, genes, uids, flanking, 
        label, lock, log_folder, sys_params, covs, bounds=[], 
        SNP_thresh=1000, save_file=False, data_fname=''):
    """Function to create a dataset from the raw genotype and phenotype file. 
    The dataset is saved as a csv file.

    Parameters
    ----------
    genotypes : xarray 
        Combined PLINK bed, bim and fam files
    phen_cov : Pandas.DataFrame
        Dataframe containing covariate and phenotype information
    chroms : list
        Chromosome numbers of the genes
    genes : list
        Symbols of the genes to create the dataset for
    uids : list
        Lis t of NCBI gene UIDs
    flanking : list
        Number of flanking bps to consider on each side of the genes
    label : str
        Phenotype or label used for prediction
    lock : multiprocessing.Manager.Lock
        Lock object to avoid concurrent access issues
    log_folder : str
        Path to log folder to record dataset errors or logs
    sys_params : dict
        Dictionary containing system specific parameters - necessary
        foder paths.
    covs : list
        List of covariates
    bounds : list, optional
        The start and end positions of the genes, by default []
    SNP_thresh : int, optional
        Maximum number of SNPs. Genes with SNPs greater 
        than this will be dropped, by default 1000
    save_file : bool, optional
        Controls if dataframe will be saved or not, by default False
    data_fname : str, optional
        Name to use while saving the final dataset file. Relevant only
        if save_file is True, by default ''
        
    Returns
    -------
    tuple
        Tuple of 2 objects:
        0 - Final data array (Pandas.DataFrame), None in the case of an 
        error
        1 - List of SNPS in the data (list), None in the case of an 
        error
    """
    covariates = copy.deepcopy(covs)
    col_missing = True
    snps = None
    final_data_array = None
    gen_info_file = False
    # Load data file if saved file exists
    if os.path.isfile(data_fname):
        final_data_array = pd.read_csv(data_fname)
        cols = list(final_data_array.columns)
        snps = [c for c in cols if 'rs' in c]
        num_snps = len(snps)

        if num_snps > SNP_thresh:
            return None, None

        # Check if file exists but has missing columns (covariates or label)
        if set(covariates).issubset(set(final_data_array.columns)) and (label in final_data_array.columns):
            col_missing = False
            final_data_array = final_data_array[
                cols[:1+num_snps]+covariates+[label,]]
            
        if col_missing:
            cols_to_add = list(
                set(covariates).union(
                set(snps)).union(
                set([label,])).difference(
                set(final_data_array.columns)))
                
            vprint("Adding missing covariates: \n{}".format(cols_to_add))
            final_data_array = add_missing_covariates(final_data_array, 
                phen_cov, cols_to_add)
            if save_file:
                final_data_array.to_csv(data_fname, index=False)
            final_data_array = final_data_array[
                cols[:1+num_snps]+covariates+[label,]]
            gen_info_file = True

    else:
        # Create data if file does not exist
        if not bounds:
            gene_bounds, genes, chroms, uids = get_gene_bounds(
                genes, uids)
        else:
            gene_bounds = bounds
        # print(covariates)
        final_data = generate_final_matrix(
            genotype, phen_cov, covariates, label, gene_bounds, flanking, 
            SNP_thresh)

        if isinstance(final_data, int):
            with open('{}/dataset_logs.txt'.format(log_folder), 'a') as f:
                f.write('{} Has {} SNPs, aborting!\n'.format(
                    '_'.join(genes), final_data))
            return None, None
        else:
            final_data_array, snps = final_data

        if save_file:
            train_ids = pd.read_csv(sys_params['TRAIN_IDS_PATH'])
            train_ids = train_ids.astype({'iid':int})

            test_ids = pd.read_csv(sys_params['TEST_IDS_PATH'])
            test_ids = test_ids.astype({'iid':int})

            id_df = pd.concat((train_ids, test_ids))
            ids = id_df['iid'].values
            labs = id_df[label].values

            final_data_array = final_data_array.astype({'iid':int})
            final_data_array.set_index('iid', inplace=True, drop=False)
            final_data_array = final_data_array.loc[ids]
            final_data_array.loc[ids, label] = labs
            final_data_array.to_csv(data_fname, index=False)

        gen_info_file = True

    if gen_info_file:
        info_fname = '{}/dataset_info.csv'.format(log_folder)
        covariates = copy.deepcopy(covs)
        extra_info = {
            'Genes': '_'.join(genes),
            'Chromosome': '_'.join([str(c) for c in chroms]),
            'Flanking_BPs': str(flanking),
            'Phenotype': label,
            'SNPs': snps,
            'Covariates':covariates,
            'Timestamp':str(datetime.datetime.now())
        }
        gen_dataset_info(final_data_array, info_fname, extra_info, lock)

    return final_data_array, snps

def generate_final_matrix(genotype, phen_cov, covariates, label_field,
                          gene_bounds, bp_buffer, SNP_thresh=1000):
    """
    Function that takes in genotype and phenotype + covariates data and 
    generates the final aggregate data matrix.
    
    Parameters
    ----------
    genotype : xarray
        PLINK genotype array containing bim, fam and bed information
    phen_cov : Pandas.DataFrame
        Phenotype and covariate information from UKB txt file
    covariates : list
        List of covariates to select
    label_field : str
        Field in the phen_cov dataframe to use as the prediction label 
    gene_bounds : ndarray 
        list of bounds for genes
    bp_buffer : list
        Base pair buffer around gene
    SNP_thresh : int
        Maximum number of SNPs. Genes with SNPs greater than this will 
        be dropped, by default 1000

    Returns
    -------
    tuple or int
        If no issues with number of SNPs, a tuple of 2 objects:
            0 - Combined data with relevant SNPs, covariates label 
            (Pandas.DataFrame)
            1 - List of SNPs in the dataset (list)
        
        If number of snps exceeds SNP_thresh or is 0, then returns
        number of SNPs    
    """

    vprint('{0:<60}{1}'.format('Raw genotype:', genotype.shape))
    vprint('{0:<60}{1}'.format('Raw phenotype:', phen_cov.shape))

    # 0. Remove indels from genotype array
    snp_ind = list(map(lambda x: 'rs' in x, np.array(genotype['snp'])))
    genotype = genotype[:, np.where(snp_ind)[0]]
    vprint('{0:<60}{1}'.format('After removing indels (genotype):', genotype.shape))

    # 1. Remove SNPs outside the gene of interest
    filt_gen = filter_SNP(genotype, gene_bounds, bp_buffer, SNP_thresh)
    if isinstance(filt_gen, int):
        return filt_gen
    vprint('{0:<60}{1}'.format('After removing SNPs', filt_gen.shape))
    
    # 2. Remove covarite columns that are not needed 
    if label_field not in covariates:
        covariates.append(label_field)
    if 'iid' not in covariates:
        covariates.insert(0, 'iid')
    filt_phen = phen_cov[covariates]

    edu_col = 'f.6138'
    edu_col_miss = 'f.6138_missing'
    if edu_col in covariates:
        # Replace -3, -7 with NaN in f.6138
        filt_phen.loc[filt_phen[edu_col] == -3, edu_col] = np.nan
        filt_phen.loc[filt_phen[edu_col] == -7, edu_col] = np.nan
        
        # Replace NaN with 1.0 in f.6138_missing i.e mark NaN as missing
        filt_phen.loc[filt_phen[edu_col_miss].isnull(), edu_col_miss] = 1.0

    filt_phen = filt_phen.loc[np.array(filt_gen['iid'], dtype=np.int)]
    vprint('{0:<60}{1}'.format('After removing covariates', filt_phen.shape))
    
    # Make sure gen and phen iids are the same before combining
    assert len(np.asarray(filt_phen.index.values, dtype=np.int)) == len(np.asarray(filt_gen['iid'], dtype=np.int))
    assert (np.asarray(filt_phen.index.values, dtype=np.int) == np.asarray(filt_gen['iid'], dtype=np.int)).all()

    # 3. Combine gen phen matrices
    agg_matrix = filt_gen.copy(deep=True)
    new_coords = {}
    for pc in filt_phen.columns:
        if pc == 'iid':
            continue
        new_coords[pc] = ('sample', filt_phen[pc])
    agg_matrix = agg_matrix.assign_coords(new_coords)
    agg_matrix[label_field] = agg_matrix[label_field].astype(float)

    # 4. Convert xarray to numpy
    data_matrix = np.zeros((agg_matrix.values.shape[0],
                            agg_matrix.values.shape[1]+len(covariates)))
    k = agg_matrix.values.shape[1]
    data_matrix[:, 1:k+1] = agg_matrix.values #SNPs
    data_matrix[:, 0] = agg_matrix['iid']
    for i, cov in enumerate(covariates):
        if cov == 'iid':
            continue
        data_matrix[:, k+i] = agg_matrix[cov]
    vprint('{0:<60}{1}'.format('After combining gen and phen', data_matrix.shape))

    # 5. Generate pandas dataframe
    snps = np.array(agg_matrix['snp'])
    column_names = np.hstack((['iid'], snps, covariates[1:]))
    final_mat = pd.DataFrame(data=data_matrix, columns=column_names)
    
    # 6. Drop iids with missing age and sex
    #    NOTE: This does not take care of missing values in the label
    #    field or additional covariates, so make sure that is taken care 
    #    of when selecting test and train sets and preparing data for NN
    final_mat.dropna(subset=['f.31.0.0', 'f.21003.0.0'], inplace=True)
    vprint('{0:<60}{1}'.format('After removing iids with missing covariates:', 
        final_mat.shape))
    
    # 7. Drop iids with more than 5% SNPs missing
    # final_mat.dropna(subset=snps, thresh=np.ceil(0.95*len(snps)), inplace=True)
    # vprint('{0:<60}{1}'.format('After removing iids >5% NaN snps:', final_mat.shape))
    
    # 8. Drop SNPs with more than 5% iids missing
    # final_mat.dropna(axis='columns', thresh=np.ceil(0.95*len(final_mat)), inplace=True)
    # vprint('{0:<60}{1}'.format('After removing SNPs >5% NaN iids:', final_mat.shape))
    
    # 9. Remove duplicated SNPs
    final_mat = final_mat.loc[:,~final_mat.columns.duplicated()]
    vprint('{0:<60}{1}'.format('After removing duplicated SNPs:', final_mat.shape))    
    num_snps = len(final_mat.columns) - len(covariates)
    if num_snps == 0:
        final_mat = None
    else:
        snps = final_mat.columns[1:1+num_snps]
    # print(final_mat.columns)
    return final_mat, snps

def add_missing_covariates(data_matrix, phen_cov, cols):
    """Function to additional data columns to an existing gene's data
    file. 

    Parameters
    ----------
    data_matrix : Pandas.DataFrame
        Gene's data, must contain the iid field
    phen_cov : Pandas.DataFrame
        Phenotype covariate matrix containing data for additional
        columns. Must contain the iid field
    cols : list
        List of column names to be included from phen_cov into 
        data_matrix

    Returns
    -------
    Pandas.DataFrame
        New dataframe with added data columns.
    """

    phen_cov.set_index('iid', drop=False, inplace=True)
    data_matrix.set_index('iid', drop=False, inplace=True)
    data_matrix.loc[data_matrix.index, cols] = phen_cov.loc[
        data_matrix.index, cols]
    
    return data_matrix

def get_gene_bounds(genes, uids):
    """
    Function to extract information about bounds of a list of genes.

    Parameters
    ----------
    genes : list or None
        Gene symbols. If symbol is not known, pass as None and uids will be
        used to extract the base pair boundaries. One of genes or uids
        must be non None
    uids : list or None
        Gene UIDs. If uids are not known, genes will be used to extract
        the base pair boundaries. One of genes or uids must be non None

    Returns
    -------
    tuple
        Tuple of 4 lists
        0 - [[start, end],] base pair positions of the genes
        1 - List of gene symbols
        2 - Chromosomes of the genes
        3 - UIDs of the genes
    """

    genes_df = pd.read_csv('/home/upamanyu/GWASOnSteroids/GWASNN/datatables/genes.csv')
    genes_df.drop_duplicates(['symbol'], inplace=True)
    if genes is not None:
        gene_rows = genes_df.loc[genes_df['symbol'].isin(genes)]
    else:
        gene_rows = genes_df.loc[genes_df['id'].isin(uids)]
    
    chroms_df = gene_rows['chrom'].to_list()
    uids_df = gene_rows['id'].to_list()
    symbols_df = gene_rows['symbol'].to_list()
    
    gene_bounds = gene_rows.apply(
        lambda x: [x['start'], x['end']], axis=1).to_list()
        
    return gene_bounds, symbols_df, chroms_df, uids_df

def filter_SNP(genotype, gene_bounds, buffer, SNP_thresh=1000):
    """
    Function to filter PLINK data to only include SNPs in the chosen GENES.

    Parameters
    ----------
    genotype : xarray
        PLINK data
    gene_bounds : ndarray
        List of gene start and end positions
    buffer : list 
        Buffer value to include around a gene's boundaries to consider 
        SNPs outside a gene but in close proximity to it.
    SNP_thresh : int, optional
        Maximum number of SNPs. Genes with SNPs greater 
        than this will be dropped, by default 1000
    
    Returns
    -------
    xarray
        Filtered genotype array

    """
    temp = genotype.copy(deep=True)
    valid_snp_indices = np.array([], dtype=np.int32)
    
    for i, bounds in enumerate(gene_bounds):
        geq_indices = np.array(temp['pos'] >= (bounds[0]-buffer[i]))
        leq_indices = np.array(temp['pos'] <= (bounds[1]+buffer[i]))

        geq_2500 = np.array(temp['pos'] >= (bounds[0]-2500))
        leq_2500 = np.array(temp['pos'] <= (bounds[1]+2500))

        final_indices = geq_indices & leq_indices
        # indices_2500 = geq_2500 & leq_2500
        # final_indices = final_indices ^ indices_2500

        valid_snp_indices = np.append(
            valid_snp_indices, np.nonzero(final_indices)[0])
    
    if len(valid_snp_indices) > SNP_thresh or len(valid_snp_indices) == 0:
        return len(valid_snp_indices)
    else:
        temp = temp[:, valid_snp_indices]
        return temp

def remove_indel_multiallelic(genotype):
    """Removes all loci which represent indels or multiallelic snps. 

    Parameters
    ----------
    genotype : xarray
        Genotype xarray loaded using pandas-plink.

    Returns
    -------
    genotype : xarray
        Genotype xarray after removing the indels and multiallelic snps.
    """
    
    vprint('{0:<60}{1}'.format('Before removing indels and multi-allelic (genotype):', genotype.shape))
    snp_ind = np.array(list(map(lambda x: 'rs' in x, np.array(genotype['snp']))), dtype=bool)

    u_snps, u_inv, u_cnts = np.unique(np.array(genotype['snp']), 
        return_inverse=True, return_counts=True)
    u_snps = u_snps[u_inv]
    u_cnts = u_cnts[u_inv]
    
    snp_mask = np.where(u_cnts==1, 
                        np.ones(len(snp_ind), dtype=bool), 
                        np.zeros(len(snp_ind), dtype=bool))
    snp_mask = snp_mask & snp_ind

    genotype = genotype[:, snp_mask]
    vprint('{0:<60}{1}'.format('After removing indels and multi-allelic (genotype):', genotype.shape))

    # Make sure that all the positions are unique as well
    pos = np.array(genotype['pos'])
    upos = np.unique(pos)
    assert len(pos) == len(upos)
    
    # Set variant coordinate to snp names
    genotype['variant'] = genotype['snp']

    return genotype

def gen_dataset_info(data, fname, extra_info, lock=None):
    """Function to generate information about a dataset and write it into a 
    csv file.
    
    Parameters
    ----------
    data : Pandas.DataFrame
        The data matrix
    fname : str
        Name of the csv file to save into
    extra_info : dict
        Dictionary of extra information to add to the info file for the 
        data
    lock : multiprocessing.Manager.Lock, optional
        The lock object used to acquire access to the csv file when 
        creating multiple datasets in parallel, by default None

    """
    num_case = len(np.where(data.iloc[:, -1] == 1)[0])
    num_control = len(np.where(data.iloc[:, -1] == 0)[0])
    
    row = {
        'Time':extra_info['Timestamp'],
        'Gene':extra_info['Genes'],
        'Chrom':extra_info['Chromosome'],
        '#Rows':data.shape[0],
        '#Cols':data.shape[1],
        '#SNPs':len(extra_info['SNPs']),
        '#Cov': len(extra_info['Covariates']),
        '#Case':num_case,
        '#Control':num_control,
        'Phenotype': extra_info['Phenotype'],
        'Flanking_BPs':extra_info['Flanking_BPs']
    }

    if lock is not None:
        lock.acquire()
    with open(fname, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
    if lock is not None:
        lock.release()

def get_balanced_data(data, label_col, oversampling_coeff=0.0, balance=1.0):
    """Function to return a balanced version of the case-control dataset.

    Parameters
    ----------
    data : Pandas.DataFrame
        CSV data file
    label_col : str
        Column in data that corresponds to the label to balance for
    oversampling_coeff : float, optional
        The percentage to oversample the minority class, by default 0.0
    balance: float, optional
        Desired ratio of control:case in the balanced data, 
        by default 1.0

    Returns 
    -------
    Pandas.DataFrame
        Balanced data of the same structure

    """
    case = data.loc[data[label_col] == 1]
    control = data.loc[data[label_col] == 0]
    
    vprint('Case count: {}'.format(case.shape))
    vprint('Case count: {}'.format(control.shape))

    # Compute required counts for case and control in balanced data
    major = case if case.shape[0] > control.shape[0] else control
    minor = case if case.shape[0] < control.shape[0] else control
    minor_cnt = int(minor.shape[0] * (1+oversampling_coeff))
    major_cnt = int(minor_cnt * balance)

    # Use sklearn resample for balancing data
    major_indices = resample(list(major.index), n_samples=major_cnt, 
                            replace=False, random_state=104)
    minor_indices = resample(list(minor.index), n_samples=minor.shape[0], 
                            replace=False, random_state=104)
    if oversampling_coeff != 0:
        oversampled = resample(minor.index, n_samples=minor_cnt-minor.shape[0], 
                            replace=False, random_state=104)
        minor_indices.extend(oversampled)

    balanced_data = pd.concat([major.loc[major_indices], minor.loc[minor_indices]])
    vprint('Case count: {}'.format(balanced_data[balanced_data[label_col] == 1].shape[0]))
    vprint('Control count: {}'.format(balanced_data[balanced_data[label_col] == 0].shape[0]))
    
    return balanced_data

def balance_by_agesex(data, label_col):
    """Function to return a 1:1 balanced version of a dataset. Balancing is 
    done to ensure the same resultant distribution for age and sex in 
    case and control.

    Parameters
    ----------
    data : Pandas.DataFrame
        CSV data file
    label_col : str
        Column in data that corresponds to the label to balance for

    Returns 
    -------
    Pandas.DataFrame
        Balanced data of the same structure

    """

    # f.21003.0.0 - Age, f.31.0.0 - Sex 
    x, y = np.unique(data[['f.21003.0.0', 'f.31.0.0']], axis=0, 
        return_counts=True)
    case_age = {'{},{}'.format(a[0],a[1]):0 for a in x}

    # Find the frequency of different ages in cases
    x, y = np.unique(data.loc[data[label_col]==1][['f.21003.0.0', 'f.31.0.0']], 
                        axis=0, return_counts=True)
    for xi, yi in zip(x,y):
        case_age['{},{}'.format(xi[0],xi[1])] = yi

    # Randomly sample controls from same distribution of age and sex as in case
    cases = data.loc[data[label_col]==1]
    conts = pd.DataFrame()
    for ca in case_age.keys():
        a, s = float(ca.split(',')[0]), float(ca.split(',')[1])
        tmp_df = data.loc[(data[label_col] == 0) & 
                            (data['f.21003.0.0'] == a) & 
                            (data['f.31.0.0'] == s)]
        if len(tmp_df) == 0:
            continue
        tmp_df = tmp_df.sample(case_age[ca], replace=False, random_state=7639)
        conts = conts.append(tmp_df)
        del(tmp_df)

    balanced_data = cases.append(conts)
    vprint('Case count: {}'.format
            (balanced_data[balanced_data[label_col] == 1].shape[0]))
    vprint('Control count: {}'.format
            (balanced_data[balanced_data[label_col] == 0].shape[0]))
    
    return balanced_data

def balance_and_fillna(data, label, na_val=-1, over_coeff=0.0, balance=1.0):
    """Function to fill missing SNP values with -1 and balance 
    cases and controls.

    Parameters
    ----------
    data : Pandas.DataFrame
        CSV data file
    label : str
        Column in data that corresponds to the label to balance for
    na_val : int, float, optional
        The percentage to oversample the minority class, by default -1
    over_coeff : float, optional
        The percentage to oversample the minority class, by default 0.0
    balance : float, optional
        Desired ratio of control:case in the balanced data, 
        by default 1.0

    Returns 
    -------
    Pandas.DataFrame
        Balanced data of the same structure
    """
    
    # Fill mising SNPs (NaN) with -1
    data.fillna(na_val, inplace=True)
    # print('Unbalanaced data size: {}'.format(data.shape))
    
    # balanced_data = get_balanced_data(data, label, over_coeff, balance)
    # print('Balanced data size: {}'.format(balanced_data.shape))

    # Balance and ensure same distribution for age and sex
    balanced_data = balance_by_agesex(data, label)
    # print('Balanced data size: {}'.format(balanced_data.shape))
    
    return balanced_data

def make_one_hot(X, num_cats):
    """Convert data to one-hot.

    Parameters
    ----------
    X : torch.Tensor
        Data tensor
    num_cats : int
        Number of categories

    Returns
    -------
    torch.Tensor
        Data tensor in one-hot encoded form
    """
    one_hot = torch.nn.functional.one_hot(X, num_cats)
    return one_hot
    