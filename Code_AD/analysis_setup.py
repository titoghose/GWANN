import sys
sys.path.append('.')

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np 
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from GWANN.dataset_utils import *
from GWANN.models import *
from GWANN.utils import *

def find_all_ids(param_folder, phen_cov_path):
    """From all possible indidividuals in the UKBB data, generate 1:1 
    case:control split and save all ids to a file. Do this for Maternal
    and Paternal histories.

    Parameters
    ----------
    param_folder : str
        Path to the folder containing experiment parameters or the
        folder where all additional parameter files should be saved.
    phen_cov_path : str
        Path to the file containing the covariates and phenotype
        information for all individuals in the cohort.    
    """
    
    geno_id_path = '{}/geno_ids.csv'.format(param_folder)
    geno_ids = pd.read_csv(geno_id_path)['ID1'].values

    ad_diag_path = '{}/AD_Diagnosis.csv'.format(param_folder)
    ad_diag = pd.read_csv(ad_diag_path)['ID1'].values

    neuro_diag_path = '{}/Neuro_Diagnosis.csv'.format(param_folder)
    neuro_diag = pd.read_csv(neuro_diag_path)['ID1'].values
    
    df = pd.read_csv(phen_cov_path, sep=' ')
    df.set_index('ID_1', drop=False, inplace=True)
    df = df.loc[df.index.isin(geno_ids)]
    print('Shape after retaining only those iids in genotype file: {}'.format(df.shape))

    # Remove people with AD diagnosis but in the FH control set 
    df = df.loc[~(df.index.isin(ad_diag) & 
        ((df['MATERNAL_MARIONI'] == 0) | (df['PATERNAL_MARIONI'] == 0)))]
    
    df.dropna(subset=['f.31.0.0', 'f.21003.0.0'], inplace=True)
    print('Shape after dropping missing age and sex: {}'.format(df.shape))
    
    df = df.loc[~((df['MATERNAL_MARIONI'] == 1) & (df['PATERNAL_MARIONI'] == 1))]
    print('Shape after removing people with both mat and pat history: {}'.format(df.shape))
    old_len = df.shape[0]
    
    df = df.loc[~(df.index.isin(neuro_diag) & 
        ((df['MATERNAL_MARIONI'] == 0) | (df['PATERNAL_MARIONI'] == 0)))]
    new_len = df.shape[0]
    print('Number of Neuro diagnosed removed: {}'.format(old_len-new_len))
    
    # Group ages
    new_ages = group_ages(df['f.21003.0.0'].values, 3)
    df['old_ages'] = df['f.21003.0.0'].values
    df['f.21003.0.0'] = new_ages
    print('Number of unique age groups: {}'.format(
        np.unique(df['f.21003.0.0'].values)))

    # Ensure that MATERNAL_MARIONI dataset is created first
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        print('\nTrain-test split for: {}'.format(label))
        all_ids_path = '{}/all_ids_{}.csv'.format(param_folder, label)
        train_ids_path = '{}/train_ids_{}.csv'.format(param_folder, label)
        test_ids_path = '{}/test_ids_{}.csv'.format(param_folder, label)
        # if os.path.isfile(test_ids_path):
        #     print('ID files exist')
        #     continue
        
        lab_df = df.copy()
        if label == 'PATERNAL_MARIONI':
            mat_test = pd.read_csv(
                '{}/test_ids_MATERNAL_MARIONI.csv'.format(
                    param_folder)).iloc[:, 0].to_list()
            mat_train = pd.read_csv(
                '{}/train_ids_MATERNAL_MARIONI.csv'.format(
                    param_folder)).iloc[:, 0].to_list()
            lab_df = lab_df.loc[~lab_df.index.isin(mat_test+mat_train)]
            assert len(lab_df.loc[lab_df.index.isin(mat_test+mat_train)]) == 0
        else:
            lab_df = lab_df.loc[~(lab_df['PATERNAL_MARIONI'] == 1)]

        # Get controls balanced by age and sex
        b_df = balance_by_agesex(lab_df, label)
        print('Final df size, cases, controls: {} {} {} {}'.format(
            b_df.shape[0], 
            b_df.loc[b_df[label] == 1].shape[0],
            b_df.loc[b_df[label] == 0].shape[0],
            b_df.loc[b_df[label].isna()].shape[0]))
        b_df = b_df.rename(columns={'ID_1':'iid'})
        b_df[['iid', label]].to_csv(all_ids_path, index=False)
        
        sss = StratifiedShuffleSplit(1, test_size=0.15, random_state=1933)
        for tr, te in sss.split(b_df.values, (b_df[label].values+1)*(b_df['f.21003.0.0'].values+1+2)):
            test_df = b_df.iloc[te]
            train_df = b_df.iloc[tr]
        
        print('Final train_df size, cases, controls: {} {} {}'.format(
            train_df.shape[0], 
            train_df.loc[train_df[label] == 1].shape[0],
            train_df.loc[train_df[label] == 0].shape[0]))

        print('Final test_df size, cases, controls: {} {} {}'.format(
            test_df.shape[0], 
            test_df.loc[test_df[label] == 1].shape[0],
            test_df.loc[test_df[label] == 0].shape[0]))

        test_df = test_df.rename(columns={'ID_1':'iid'})
        test_df[['iid', label]].to_csv(test_ids_path, index=False)
        
        train_df = train_df.rename(columns={'ID_1':'iid'})
        train_df[['iid', label]].to_csv(train_ids_path, index=False)
        
        D, p = stats.ks_2samp(train_df['old_ages'], test_df['old_ages'])
        print("Test and train ages KS test: p={}".format(p))
        plt.hist(train_df['old_ages'], density=True, alpha=0.5, label='Train')
        plt.hist(test_df['old_ages'], density=True, alpha=0.5, label='Test')
        plt.legend()
        plt.savefig('{}/age_dist_{}.png'.format(param_folder, label))
        plt.close()

        D, p = stats.ks_2samp(train_df['f.31.0.0'], test_df['f.31.0.0'])
        print("Test and train sex KS test: p={}".format(p))
        plt.hist(train_df['f.31.0.0'], density=True, alpha=0.5, label='Train')
        plt.hist(test_df['f.31.0.0'], density=True, alpha=0.5, label='Test')
        plt.legend()
        plt.savefig('{}/sex_dist_{}.png'.format(param_folder, label))
        plt.close()

def create_groups(param_folder, phen_cov_path, grp_size,
            train_oversample=10, test_oversample=10):
    """Convert data arrays to grouped data arrays after balancing as
    best as possible for age and sex.

    Parameters
    ----------
    param_folder : str
        Path to the folder containing experiment parameters or the
        folder where all additional parameter files should be saved.
    phen_cov_path : str
        Path to the file containing the covariates and phenotype
        information for all individuals in the cohort.    
    grp_size : int
        Size of groups
    train_oversample : int, optional
        Factor to oversample all train data samples by before forming into
        groups, by default 10
    test_oversample : int, optional
        Factor to oversample all test data samples by before forming into
        groups, by default 10

    Returns
    -------
    dict of tuples
        Each tuple of 4 ndarrays:
        0 - Grouped train set ids
        1 - Train labels
        2 - Grouped test set ids
        3 - Test labels
    """
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        print('\nGrouping ids for for: {}'.format(label))
        
        group_id_path = '{}/group_ids_{}.npz'.format(param_folder, label)
        if os.path.isfile(group_id_path):
            print('Group ids file exists')
            continue
        
        df = pd.read_csv(phen_cov_path, sep=' ')
        df.set_index('ID_1', drop=False, inplace=True)
        
        train_ids_path = '{}/train_ids_{}.csv'.format(param_folder, label)
        train_ids = pd.read_csv(train_ids_path)['iid'].values
        test_ids_path = '{}/test_ids_{}.csv'.format(param_folder, label)
        test_ids = pd.read_csv(test_ids_path)['iid'].values
        
        X = df.loc[train_ids]
        y = df.loc[train_ids][label].values
        Xt = df.loc[test_ids]
        yt = df.loc[test_ids][label].values

        grp_ids = []
        grp_labels = []
        for X_, y_, over in [(X, y, train_oversample), (Xt, yt, test_oversample)]:
            
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
            sex = X_['f.31.0.0'].values
            age = X_['f.21003.0.0'].values
            vprint('Ages : {}'.format(np.unique(age)))
            age = group_ages(age, num_grps=3)
            for j, idxs in enumerate([case, cont]):
                n_splits = round(len(idxs)/grp_size)
                X_idxs = X_.iloc[idxs, :]
                sex_idxs = sex[idxs]
                age_idxs = age[idxs]

                # Combine age groups and sex to form stratified groups
                age_sex = np.add(age_idxs, sex_idxs*3)
                vprint('Age groups: {}'.format(np.unique(age_idxs)))
                vprint('Sex: {}'.format(np.unique(sex_idxs)))
                vprint('Unique age_sex groups: {}'.format(np.unique(age_sex)))
                
                skf = StratifiedKFold(
                            n_splits=n_splits, 
                            shuffle=True, 
                            random_state=4231)
                for _, ind in skf.split(np.zeros(len(idxs)), age_sex):
                    Xg[j].append(X_idxs.iloc[ind].index.values)
                    
            grp_ids.append(np.concatenate((Xg[0], Xg[1])))
            grp_labels.append(np.concatenate((np.ones(len(Xg[0])), np.zeros(len(Xg[1])))))
    
        np.savez(group_id_path,
            train_grps=grp_ids[0],
            train_grp_labels=grp_labels[0],
            test_grps=grp_ids[1],
            test_grp_labels=grp_labels[1])

if __name__ == '__main__':
    
    # 1. Find all train and test ids
    find_all_ids(
        param_folder='Code_AD/params', 
        phen_cov_path='/mnt/sdc/UKBB_2/Variables_UKB.txt')

    # 2. Group ids to enable "group-training"
    create_groups(
        param_folder='Code_AD/params', 
        phen_cov_path='/mnt/sdc/UKBB_2/Variables_UKB.txt',
        grp_size=10
    )