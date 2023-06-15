import sys
from matplotlib import pyplot as plt

sys.path.append('/home/upamanyu/GWANN')

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import run_genes
import dummy_genes

def sensitivity_1(chroms:list):
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'
    
    # mat_df = pd.read_csv(f'{param_folder}/Annotated_NoDup_MatAD_NN_GWAS.csv')
    # pat_df = pd.read_csv(f'{param_folder}/Annotated_NoDup_PatAD_NN_GWAS.csv')
    # meta_df = pd.read_csv(f'{param_folder}/Annotated_AD_Meta.csv')
    
    # gdf = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv')
    # marioni_meta_and_common_hits = ['CR1', 'PICALM', 'ZNF232', 'BIN1', 'NIFKP9', 
    #                 'APP', 'PILRA', 'ZCWPW1', 'NYAP1', 'CLU', 'SIGLEC11', 
    #                 'ABCA1', 'ABCA7']
    # marioni_df = gdf.loc[gdf['symbol'].isin(marioni_meta_and_common_hits)]
    # marioni_df.rename(columns={'symbol':'Gene', 'chrom':'Chrom'}, inplace=True)
    # marioni_df['P'] = 0

    # labels = {'PATERNAL_MARIONI':marioni_df, 
    #           'MATERNAL_MARIONI':marioni_df}
    # for label, df in labels.items():
    #     hits = df.loc[df['P'] < (0.05/73310)].astype({'Chrom':str})
    #     print(label)
    #     for chrom, idx in hits.groupby('Chrom').groups.items():
    #         if not str(chrom) in chroms:
    #             continue
    #         glist = hits.loc[idx]['Gene'].to_list()
    #         print(chrom, glist)
    #         run_genes.create_csv_data(label=label, param_folder=param_folder, 
    #                         chrom=chrom, glist=glist, split=True)

    gpu_list = list(np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3))
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        # run_genes.model_pipeline(exp_name='Sens1', label=label, 
        #             param_folder=param_folder, gpu_list=gpu_list)
        # dummy_genes.model_pipeline(exp_name='Sens1Dummy', label=label, 
        #             param_folder=param_folder, gpu_list=gpu_list)
    
        summ_df = pd.read_csv(f'./results_sensitivity1/{label}_METAL_genes_inp.csv', sep=' ')
        summ_df.set_index('Gene', inplace=True)
        old_summ_df = pd.read_csv(f'/home/upamanyu/GWASOnSteroids/Results/Annotated_NoDup_{label[0]+label[1:3].lower()}AD_NN_GWAS.csv')
        old_summ_df.set_index('Gene', inplace=True)
        summ_df['old_Acc'] = old_summ_df.loc[summ_df.index]['Acc'].values
        fig, ax = plt.subplots(1, 1)
        summ_df.drop(index=['APOE', 'APOC1', 'TOMM40', 'BCAM'], inplace=True)
        print(label, spearmanr(summ_df['Acc'], summ_df['old_Acc']))
        sns.scatterplot(data=summ_df, x='old_Acc', y='Acc', size=5, ax=ax, legend=False)
        fig.savefig(f'./results_sensitivity1/{label}_oldvsnewRes.png', dpi=100)
        plt.close()
    
    genes_df = pd.read_csv(f'./results_sensitivity1/Parental_AD_genes_METAL1.csv', sep='\t')

    wins_df = pd.read_csv(f'./results_sensitivity1/Parental_AD_wins_METAL1.csv', sep='\t')
    wins_df['MarkerName'] = wins_df['MarkerName'].apply(lambda x:x.split('_')[0]).values
    wins_df.sort_values(['MarkerName', 'P-value'], inplace=True)
    wins_df.drop_duplicates(['MarkerName'], inplace=True)
    
    for key, meta_summ_df in {'genes':genes_df, 'wins':wins_df}.items():
        meta_summ_df.set_index('MarkerName', inplace=True)
        old_meta_summ_df = pd.read_csv(f'/home/upamanyu/GWASOnSteroids/Results/Annotated_AD_Meta.csv')
        old_meta_summ_df.set_index('Gene', inplace=True)
        meta_summ_df['P'] = meta_summ_df['P-value']
        meta_summ_df['P'] = -np.log10(meta_summ_df['P'].values)
        meta_summ_df['old_P'] = old_meta_summ_df.loc[meta_summ_df.index]['P'].values
        meta_summ_df['old_P'] = -np.log10(meta_summ_df['old_P'].values)
        meta_summ_df.drop(index=['APOE', 'APOC1', 'TOMM40', 'BCAM'], inplace=True)
        sns.scatterplot(data=meta_summ_df, x='old_P', y='P', size=5, legend=False)
        plt.savefig(f'./results_sensitivity1/Parental_meta_{key}_oldvsnewRes.png', dpi=100)
        plt.close()

def sensitivity_1_2(chroms:list):
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'
    gpu_list = list(np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        # run_genes.model_pipeline(exp_name='Sens1.2', label=label, 
        #             param_folder=param_folder, gpu_list=gpu_list)
        dummy_genes.model_pipeline(exp_name='Sens1.2Dummy', label=label, 
                    param_folder=param_folder, gpu_list=gpu_list)
        break
    
    #     summ_df = pd.read_csv(f'./results_sensitivity1/{label}_METAL_genes_inp.csv', sep=' ')
    #     summ_df.set_index('Gene', inplace=True)
    #     old_summ_df = pd.read_csv(f'/home/upamanyu/GWASOnSteroids/Results/Annotated_NoDup_{label[0]+label[1:3].lower()}AD_NN_GWAS.csv')
    #     old_summ_df.set_index('Gene', inplace=True)
    #     summ_df['old_Acc'] = old_summ_df.loc[summ_df.index]['Acc'].values
    #     fig, ax = plt.subplots(1, 1)
    #     summ_df.drop(index=['APOE', 'APOC1', 'TOMM40', 'BCAM'], inplace=True)
    #     print(label, spearmanr(summ_df['Acc'], summ_df['old_Acc']))
    #     sns.scatterplot(data=summ_df, x='old_Acc', y='Acc', size=5, ax=ax, legend=False)
    #     fig.savefig(f'./results_sensitivity1/{label}_oldvsnewRes.png', dpi=100)
    #     plt.close()
    
    # genes_df = pd.read_csv(f'./results_sensitivity1/Parental_AD_genes_METAL1.csv', sep='\t')

    # wins_df = pd.read_csv(f'./results_sensitivity1/Parental_AD_wins_METAL1.csv', sep='\t')
    # wins_df['MarkerName'] = wins_df['MarkerName'].apply(lambda x:x.split('_')[0]).values
    # wins_df.sort_values(['MarkerName', 'P-value'], inplace=True)
    # wins_df.drop_duplicates(['MarkerName'], inplace=True)
    
    # for key, meta_summ_df in {'genes':genes_df, 'wins':wins_df}.items():
    #     meta_summ_df.set_index('MarkerName', inplace=True)
    #     old_meta_summ_df = pd.read_csv(f'/home/upamanyu/GWASOnSteroids/Results/Annotated_AD_Meta.csv')
    #     old_meta_summ_df.set_index('Gene', inplace=True)
    #     meta_summ_df['P'] = meta_summ_df['P-value']
    #     meta_summ_df['P'] = -np.log10(meta_summ_df['P'].values)
    #     meta_summ_df['old_P'] = old_meta_summ_df.loc[meta_summ_df.index]['P'].values
    #     meta_summ_df['old_P'] = -np.log10(meta_summ_df['old_P'].values)
    #     meta_summ_df.drop(index=['APOE', 'APOC1', 'TOMM40', 'BCAM'], inplace=True)
    #     sns.scatterplot(data=meta_summ_df, x='old_P', y='P', size=5, legend=False)
    #     plt.savefig(f'./results_sensitivity1/Parental_meta_{key}_oldvsnewRes.png', dpi=100)
    #     plt.close()

if __name__ == '__main__':
    chroms = sys.argv[1].split(',')
    sensitivity_1_2(chroms)
