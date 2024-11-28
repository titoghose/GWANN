import argparse
import os
import sys

sys.path.append('/home/upamanyu/GWANN')

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml

from GWANN.dataset_utils import PGEN2Pandas, balance_by_agesex, group_ages, load_data
from GWANN.train_utils import infer, training_stuff


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

    pgen_prefix = f'{sys_params["RAW_BASE_FOLDER"][chrom]}/UKB_chr{chrom}'
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
                        f'Chr{chrom}_FH_AD_ChrSens8_{rseed}{rseed}_GS10_v4_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/' +
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
        if os.path.exists(f'results_non_white/{pop}_results_top100.csv'):
            df = pd.read_csv(f'results_non_white/{pop}_results_top100.csv')
            summary_df = pd.concat([df, summary_df], ignore_index=True)
        summary_df.to_csv(f'results_non_white/{pop}_results_top100.csv', index=False)

def create_cohort(pop:str, sys_params:dict, covs:list):
    """Create the cohort of cases and controls for the non-white populations.

    Parameters
    ----------
    pop : str
        Population
    params : dict
        Dictionary of the paths for various files
    covs : list
        List of covariates to be used
    """
    all_samples = pd.read_csv(f'params_non_white/{pop}_or_{pop}_British_genetic_QC.csv')
    all_samples = set(all_samples['eid'].astype(str).to_list())
    
    ad_samples = pd.read_csv(f'params_non_white/{pop}_AD_plus_FH.csv')
    ad_samples = set(ad_samples['eid'].astype(str).to_list())
    
    all_dementia_samples = pd.read_csv(f'params_non_white/{pop}_Dementia_plus_FH.csv')
    all_dementia_samples = set(all_dementia_samples['eid'].astype(str).to_list())

    other_neuro_diag_samples = pd.read_csv(f'params_non_white/Neuro_Diagnosis.csv')
    other_neuro_diag_samples = set(other_neuro_diag_samples['ID_1'].astype(str).to_list())
    
    ukb_withdrawn = pd.read_csv('params_non_white/ukb_withdrawn_04May23.csv')
    ukb_withdrawn = set(ukb_withdrawn['ID_1'].astype(str).to_list())

    ukb_variables = pd.read_csv(sys_params['PHEN_COV_PATH'], sep=' ', comment='#')
    ukb_variables['ID_1'] = ukb_variables['ID_1'].astype(str)
    ukb_variables.set_index('ID_1', inplace=True)
    
    # Cases
    # 1. Keep only AD+FH_AD samples that passed genetic QC
    # 2. Remove withdrawn samples
    print('Number of cases: ', len(ad_samples)) 
    cases = ad_samples & all_samples
    print('Number of cases after QC filter: ', len(cases)) 
    cases = cases - ukb_withdrawn
    print('Number of cases after removing withdrawn: ', len(cases))

    # Controls
    # 1. Keep all samples that passed genetic QC that do not have
    #    AD/FH_AD
    # 2. Remove all samples with all forms of dementia
    # 3. Remove samples with other neurological diagnosis
    # 4. Remove withdrawn samples
    controls = all_samples - ad_samples
    controls = controls - all_dementia_samples
    controls = controls - other_neuro_diag_samples
    print('Number of controls after removing AD, Dementia, or other neuro: ', len(controls))
    controls = controls - ukb_withdrawn
    print('Number of controls after removing withdrawn: ', len(controls))

    # Remove cases and controls with missing covariates
    sample_dict = {'cases': cases, 'controls': controls}
    for k in sample_dict.keys():
        sample_df = ukb_variables.loc[list(sample_dict[k]), covs]
        sample_df = sample_df.dropna()
        sample_dict[k] = sample_df.index.to_list()
        print(f'Number of {k} after removing those with missing covs: ', len(sample_dict[k]))

    # balance cases and controls by age and sex
    combined_df = ukb_variables.loc[sample_dict['cases'] + sample_dict['controls']].copy()
    combined_df = combined_df.filter(covs)
    combined_df['FH_AD'] = 0
    combined_df.loc[sample_dict['cases'], 'FH_AD'] = 1
    grouped_ages = group_ages(combined_df['f.21003.0.0'].values, 10)
    combined_df['old_ages'] = combined_df['f.21003.0.0'].values
    combined_df['f.21003.0.0'] = grouped_ages
    print('Number of unique age groups: {}'.format(
        np.unique(combined_df['f.21003.0.0'].values)))
    
    balanced_df = balance_by_agesex(combined_df, 'FH_AD', control_prop=1)
    print('Balanced cohort size: ', balanced_df.shape[0])

    # Save the balanced cohort
    balanced_samples = balanced_df.reset_index().filter(['ID_1', 'FH_AD'])
    balanced_samples.rename(columns={'ID_1': 'iid'}, inplace=True)
    balanced_samples.to_csv(f'params_non_white/reviewer_rerun_Sens8/{pop}_ids_FH_AD.csv', index=False)

def top_100_genes_from_white_population() -> pd.DataFrame:
    """Load the top 100 genes from the UKB white british population
    after LD pruning. 

    Returns
    -------
    pd.DataFrame
        Top 100 genes with annotations needed to run the pipeline
    """
    # Load summary file from the testing set in the white population
    # This is the file that contains information after LD pruning
    summary_df = pd.read_csv('results_Sens8_v4/LD/r20.8_pruned_gene_hits_5e-09.csv')
    summary_df = summary_df.loc[~summary_df['pruned']]
    summary_df = summary_df.sort_values('p_stat_trial_A')
    top_100 = summary_df.head(100)

    # Load gene annotation file
    gdf = pd.read_csv('../GWANN/datatables/gene_annot.csv', 
                      dtype={'chrom':str})
    gdf.set_index('symbol', inplace=True)
    gdf = gdf.loc[top_100['Gene']]
    gdf['win'] = top_100['Win'].values
    gdf.reset_index(inplace=True)
    gdf.rename(columns={'symbol': 'gene'}, inplace=True)
    gdf  = gdf.filter(['gene', 'chrom', 'win', 'start', 'end'])

    assert gdf['gene'].to_list() == top_100['Gene'].to_list()
    
    return gdf

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on the top 100 genes from the white population')
    parser.add_argument('--pop', type=str, help='Population to run inference on')
    parser.add_argument('--seeds', type=str, help='List of random seeds')
    parser.add_argument('--gpu', type=str, help='GPU id')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    pop = args.pop
    os.environ['GPU'] = args.gpu

    with open('params_non_white/reviewer_rerun_Sens8/params_FH_AD.yaml', 'r') as f:
        params = yaml.safe_load(f)

    with open('params_non_white/reviewer_rerun_Sens8/covs_FH_AD.yaml', 'r') as f:
        covs = yaml.safe_load(f)['COVARIATES']

    # create_cohort(pop, params, covs)
    
    ids = pd.read_csv(f'params_non_white/reviewer_rerun_Sens8/{pop}_ids_FH_AD.csv')
    
    # Save ids as train and test because the data loader expects it
    ids.iloc[:100, :].to_csv(f'params_non_white/reviewer_rerun_Sens8/test_ids_FH_AD.csv', 
                                index=False)
    ids.iloc[100:, :].to_csv(f'params_non_white/reviewer_rerun_Sens8/train_ids_FH_AD.csv', 
                                index=False)

    # Get top 100 gene windows from the white population
    gene_df = top_100_genes_from_white_population()
    
    # Run inference on multiple seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    results = run_inference(gene_df, params, covs, ids['iid'].astype(str).to_list(), seeds)
    
    # Clean up
    # Delete the train and test ids
    os.remove(f'params_non_white/reviewer_rerun_Sens8/test_ids_FH_AD.csv')
    os.remove(f'params_non_white/reviewer_rerun_Sens8/train_ids_FH_AD.csv')
