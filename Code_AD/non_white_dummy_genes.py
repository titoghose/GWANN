import argparse
import multiprocessing as mp
import os
import shutil
import sys

sys.path.append('/home/upamanyu/GWANN')

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml

from GWANN.dataset_utils import PGEN2Pandas, load_data
from GWANN.dummy_data import dummy_plink, merge_pgen
from GWANN.train_utils import infer, training_stuff


# Copied from non_white_validation.py with slight modifications to path
def get_data(gene_dict:dict, sys_params:dict, covs:list, ids:list) -> tuple:
    """Interface to load data for a gene.

    Parameters
    ----------
    gene_dict : dict
        Infomation about the gene
    sys_params : dict
        Dictionary of the paths for various files
    covs : list
        List of covariates to be used
    ids : list
        List of case-control ids

    Returns
    -------
    tuple
        Data tuple
    """
    gene = gene_dict['gene']
    chrom = gene_dict['chrom']
    start = gene_dict['start'] if 'start' in gene_dict else None
    end = gene_dict['end'] if 'end' in gene_dict else None
    win = gene_dict['win'] if 'win' in gene_dict else None

    pgen_prefix = f'{sys_params["RAW_BASE_FOLDER"]["Dummy"]}/Dummy_merged'
    pg2pd = PGEN2Pandas(pgen_prefix, sample_subset=ids)
        
    phen_cov = pd.read_csv(sys_params['PHEN_COV_PATH'], 
                        sep=' ', dtype={'ID_1':str}, comment='#')
    phen_cov = phen_cov.rename(columns={'ID_1':'iid'})
    phen_cov.index = phen_cov['iid']
    
    data = load_data(
                pg2pd=pg2pd, 
                phen_cov=phen_cov, 
                gene=gene, chrom=chrom, start=start, end=end, 
                buffer=2500, # Fixed buffer size that was used for training 
                label='FH_AD', 
                sys_params=sys_params, 
                covs=covs, 
                win=win, 
                save_data=False, 
                SNP_thresh=10000, 
                only_covs=False, 
                lock=None)
    
    return data

# Copied from non_white_validation.py with slight modifications to path
def run_inference(gene_df:pd.DataFrame, sys_params:dict, covs:list, ids:list, 
                    rseeds:list):
    """Run inference on the top 100 genes from the white population.

    Parameters
    ----------
    gene_df : pd.DataFrame
        List of genes to run inference on
    sys_params : dict
        Dictionary of the paths for various files
    covs : list
        List of covariates to be used
    ids : list
        List of case-control ids
    """
    gene_dicts = gene_df.to_dict(orient='records')
    for gene_dict in tqdm.tqdm(gene_dicts, desc='Gene', leave=True):
        # print('Gene: ', gene_dict['gene'])
        # print('Loading data...')
        data = get_data(gene_dict, sys_params, covs, ids)
        X, y, X_test, y_test, cw, data_cols, num_snps = data
        
        # Combine train and test data
        X = torch.cat([
            torch.from_numpy(X), 
            torch.from_numpy(X_test)], dim=0).float()
        y = torch.cat([
            torch.from_numpy(y), 
            torch.from_numpy(y_test)], dim=0).float()
        # print('X.shape: ', X.shape)
        # print('y.shape: ', y.shape)

        # Load the model
        # print('Loading model...')
        gene = gene_dict['gene']
        chrom = gene_dict['chrom']
        win = gene_dict['win']
        results = []
        for rseed in tqdm.tqdm(rseeds, desc='Seed', leave=False):
            os.environ['GROUP_SEED'] = str(rseed)
            model_path = (f'{sys_params["LOGS_BASE_FOLDER"]}/' +
                        f'FH_AD_ChrDummySens8_{rseed}{rseed}_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/' +
                        f'{gene}_{win}/{num_snps}_{gene}_{win}.pt')
            model = torch.load(model_path, map_location=torch.device(int(os.environ['GPU'])))
            # print('Loading loss function...')
            loss_fn, _, _ = training_stuff(model, damping=1.0, 
                                            class_weights=torch.Tensor([1, 1]), 
                                            lr=0.005, opt='adam')
            loss_fn = loss_fn.to(int(os.environ['GPU']))
            
            # Run inference
            # print('Running inference...')
            res = infer(X, y, model, loss_fn, int(os.environ['GPU']))
            del res['conf_mat']
            del res['roc_auc']
            res.update({'Gene': f'{gene}_{win}', 'Seed':rseed, 'Chrom':chrom, 'SNPs': num_snps})
            results.append(res)
        
        summary_df = pd.DataFrame.from_records(results)
        if os.path.exists(f'results_non_white/{pop}_results_Dummy.csv'):
            df = pd.read_csv(f'results_non_white/{pop}_results_Dummy.csv')
            summary_df = pd.concat([df, summary_df], ignore_index=True)
        summary_df.to_csv(f'results_non_white/{pop}_results_Dummy.csv', index=False)
    
def get_dummy_gene_df():
    snps_per_gene = 500
    gene_win_df = pd.DataFrame(columns=['chrom', 'gene', 'win', 'win_count', 'start', 'end'])
    gene_win_df['gene'] = [f'Dummy{i}' for i in range(1000)]
    gene_win_df['win'] = [0]*1000
    gene_win_df['start'] = [i*snps_per_gene for i in range(1000)]
    gene_win_df['end'] = [(i+1)*snps_per_gene for i in range(1000)]
    gene_win_df['end'] -= 1
    gene_win_df['chrom'] = '1'
    return gene_win_df

# Copied from dummy_genes.py with slight modifications to path
def create_dummy_pgen(param_folder:str, label:str, sys_params:dict, covs:list) -> None:
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

    ids = pd.read_csv(f'{param_folder}/all_ids_FH_AD.csv', dtype={'iid':str})['iid'].to_list()
    print(len(ids))

    out_folder = sys_params['RAW_BASE_FOLDER']['Dummy']
    
    phen_cov = pd.read_csv(sys_params['PHEN_COV_PATH'], sep=' ', comment='#',
                           dtype={'ID_1':str})
    phen_cov.rename(columns={'ID_1':'iid'}, inplace=True)
    phen_cov.set_index('iid', inplace=True)

    lock = mp.Manager().Lock()
    
    file_prefixes = []
    cnt = 0
    num_snps = 500
    dosage_freqs = [0.0, 0.02, 0.04, 0.06, 0.08]
    for num_snps in tqdm.tqdm([num_snps]*200, desc='Num_dummy_snps'):
        for dos_freq in dosage_freqs:
            file_prefix = dummy_plink(samples=ids, 
                    num_snps=num_snps, dosage_freq=dos_freq, 
                    out_folder=out_folder,
                    var_pos_offset=num_snps*cnt,
                    file_prefix=f'Dummy{cnt}')
            
            file_prefixes.append(file_prefix)
            cnt += 1

    with open('.temp_file_prefixes.txt', 'w') as f:
        f.write('\n'.join(file_prefixes))
    
    merge_pgen(pgen_prefix_file=os.path.abspath('.temp_file_prefixes.txt'),
               out_folder=out_folder,
               file_prefix=f'Dummy')

    os.remove('.temp_file_prefixes.txt')

# Copied from non_white_validation.py
def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on the top 100 genes from the white population')
    parser.add_argument('--pop', type=str, required=True, help='Population to run inference on')
    parser.add_argument('--seeds', type=str, required=True, help='List of random seeds')
    parser.add_argument('--gpu', type=str, required=True, help='GPU id')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    pop = args.pop
    os.environ['GPU'] = args.gpu

    param_folder='/home/upamanyu/GWANN/Code_AD/params_non_white/reviewer_rerun_Sens8'
    label = 'FH_AD'

    with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    with open('{}/covs_{}.yaml'.format(param_folder, label), 'r') as f:
        covs = yaml.load(f, Loader=yaml.FullLoader)['COVARIATES']
    params['RAW_BASE_FOLDER']['Dummy'] = params['RAW_BASE_FOLDER']['Dummy'] + pop

    # Dummy data creation
    # create_dummy_pgen(param_folder=param_folder, label=label, sys_params=params, covs=covs)

    # Save ids as all, train and test because the data loader expects it
    ids = pd.read_csv(f'params_non_white/reviewer_rerun_Sens8/{pop}_ids_FH_AD.csv')
    ids.to_csv('params_non_white/reviewer_rerun_Sens8/all_ids_FH_AD.csv', 
                                index=False)
    ids.iloc[:100, :].to_csv(f'params_non_white/reviewer_rerun_Sens8/test_ids_FH_AD.csv', 
                                index=False)
    ids.iloc[100:, :].to_csv(f'params_non_white/reviewer_rerun_Sens8/train_ids_FH_AD.csv', 
                                index=False)
  
    # Run inference on multiple seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    gene_df = get_dummy_gene_df()
    
    results = run_inference(gene_df, params, covs, ids['iid'].astype(str).to_list(), seeds)

    # Clean up
    # Delete the train and test ids
    os.remove(f'params_non_white/reviewer_rerun_Sens8/all_ids_FH_AD.csv')
    os.remove(f'params_non_white/reviewer_rerun_Sens8/test_ids_FH_AD.csv')
    os.remove(f'params_non_white/reviewer_rerun_Sens8/train_ids_FH_AD.csv')