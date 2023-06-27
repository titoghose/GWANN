import os
import sys

import tqdm

sys.path.append('/home/upamanyu/GWANN')

import multiprocessing as mp
import shutil
import warnings
from functools import partial
from typing import Optional

import yaml

warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import StratifiedShuffleSplit

from GWANN.dataset_utils import (balance_by_agesex, create_data_for_run,
                                 create_groups, group_ages, split)
from GWANN.utils import vprint


def filter_ids(param_folder:str, phen_cov_path:str) -> None:
    """Additional filters to remove ids that were selected during the
    original analysis to create a new set of ids for the re-run to
    address reviewer comments. Retain most of the old ids except where
    (i) individuals withdrew participation from UKB, (ii) have any
    missing values for covariates, or (iii) sample is a control but is
    in the list of AD diagnosed people.

    param_folder : str
        Path to the folder containing experiment parameters or the
        folder where all additional parameter files should be saved.
    phen_cov_path : str
        Path to the file containing the covariates and phenotype
        information for all individuals in the cohort.
    """
    extended_AD_ids = pd.read_csv(
        './params/UKB_AD_inc_c.sample', sep='\t').iloc[1:,:]['ID_1'].to_list()
    
    ukb_withdrawn_ids = pd.read_csv(
            'params/ukb_withdrawn_04May23.csv')['ID_1'].to_list()

    neuro_ids = pd.read_csv('./params/Neuro_Diagnosis.csv')['ID_1'].to_list()

    with open(f'{param_folder}/covs_MATERNAL_MARIONI.yaml', 'r') as f:
        covs = yaml.load(f, yaml.FullLoader)['COVARIATES']
    phen_cov_df = pd.read_csv(phen_cov_path, sep=' ', comment='#')
    phen_cov_df = phen_cov_df[
        ['ID_1',] + covs + ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']]
    phen_cov_df.set_index('ID_1', inplace=True, drop=False)
    phen_cov_df.dropna(subset=covs, axis=0, how='any', inplace=True)
    
    original_ids = []
    for phen in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        for split in ['train', 'test']:
            ids = pd.read_csv(f'./params/original_run/{split}_ids_{phen}.csv')['iid'].to_list()
            original_ids.extend(ids)
    
    add_df = phen_cov_df.loc[~phen_cov_df.index.isin(original_ids)]
    add_df = add_df.loc[
                (add_df['MATERNAL_MARIONI'] != 1) & 
                (add_df['PATERNAL_MARIONI'] != 1) &
                (~add_df.index.isin(neuro_ids)) &
                (~add_df.index.isin(extended_AD_ids)) &
                (~add_df.index.isin(ukb_withdrawn_ids))]

    for phen in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        for split in ['train', 'test']:
            
            vprint(f'\n\n{phen} - {split}')
            vprint('------------')
            id_df = pd.read_csv(f'./params/original_run/{split}_ids_{phen}.csv')
            id_df.set_index('iid', inplace=True, drop=False)
            vprint(id_df.groupby(phen).count().T)

            # Remove iids with missing covariate data   
            id_df = id_df.loc[id_df.index.isin(phen_cov_df.index)]
            
            # Remove controls with AD diagnosis
            id_df = id_df.loc[~((id_df[phen] == 0) & 
                                (id_df.index.isin(extended_AD_ids)))]
            
            # Remove cases and controls in the withdrawn list
            id_df = id_df.loc[~id_df.index.isin(ukb_withdrawn_ids)]
            
            vprint(f'\nAfter filters:')
            vprint(id_df.groupby(phen).count().T)
            # print(id_df.groupby(phen).count().index)

            # If n(controls) < n(cases) add controls to make ratio 1:1
            num_extra_cases = (id_df.loc[id_df[phen] == 1].shape[0] - 
                               id_df.loc[id_df[phen] == 0].shape[0])
            add_controls = add_df.loc[add_df[phen] == 0].sample(
                                        num_extra_cases, random_state=2617)
            add_df.drop(index=add_controls.index, inplace=True)
            add_controls.rename(columns={'ID_1': 'iid'}, inplace=True)
            id_df = pd.concat((id_df, add_controls[['iid', phen]]))
    
            vprint(f'\nAfter adding new controls:')
            vprint(id_df.groupby(phen).count().T)

            id_df.to_csv(f'./params/reviewer_rerun/{split}_ids_{phen}.csv', 
                         index=False)
        
        create_groups(
            label=phen,
            param_folder=param_folder, 
            phen_cov_path=phen_cov_path,
            grp_size=10
        )

def find_all_ids(param_folder:str, phen_cov_path:str) -> None:
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

def dosage_percentage():
    base = '/home/upamanyu/GWANN_data/Data_MatAD/wins'
    flist = os.listdir(base)
    dos_gt_ptg = []
    for f in tqdm.tqdm(flist[:500]):
        df = pd.read_csv(f'{base}/{f}', index_col=0)
        df = df[df.columns[:-10]]
        hard_gt = np.count_nonzero(np.isin(df.values, [0.0, 1.0, 2.0]))
        total_gt = df.shape[0]*df.shape[1]
        dos_gt_ptg.append((total_gt-hard_gt)/total_gt)
        # break
    
    sns.histplot(x=dos_gt_ptg)
    plt.savefig('dosage_percentage_dist.png', dpi=100)
    plt.close()

if __name__ == '__main__':
    
    # 1. Find all train and test ids
    # find_all_ids(
    #     param_folder='Code_AD/params', 
    #     phen_cov_path='/mnt/sdc/UKBB_2/Variables_UKB.txt')

    # Cell Reports reviewer rerun
    # filter_ids(
    #     param_folder='/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun', 
    #     phen_cov_path='/mnt/sdg/UKB/Variables_UKB.txt')
    
    # dosage_percentage()

    create_groups(
            label='MATERNAL_MARIONI',
            param_folder='/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun', 
            phen_cov_path='/mnt/sdg/UKB/Variables_UKB.txt',
            grp_size=10
        )