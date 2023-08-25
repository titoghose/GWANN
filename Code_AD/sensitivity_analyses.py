import os
import sys
from matplotlib import pyplot as plt
import yaml

sys.path.append('/home/upamanyu/GWANN')

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, rankdata
import run_genes
import dummy_genes
import cov_model
from GWANN.dataset_utils import create_groups

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

def sensitivity_1_3(chroms:list):
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'
    gpu_list = list(np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
    glist = (['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'] +
            ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
            ['ICAM3', 'ATP2C1', 'GLIS3', 'ARHGEF28']) # GWANNv1 Mat AD within bottom 20 genes that passed significance
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        # run_genes.model_pipeline(exp_name='Sens1.3', label=label, 
        #             param_folder=param_folder, gpu_list=gpu_list, 
        #             glist=glist)
        dummy_genes.model_pipeline(exp_name='Sens1.3Dummy', label=label, 
                    param_folder=param_folder, gpu_list=gpu_list)
        break

def sensitivity_1_4(chroms:list):
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'

    gdf = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv')
    
    # GWANNv1 Mat AD APOE locus genes
    # gdf = gdf.loc[gdf['symbol'].isin(['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'])]
    # gdf = gdf.loc[gdf['symbol'].isin(
    #     ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
    #     ['ICAM3', 'ATP2C1', 'PPP1R37', 'GLIS3', 'ARHGEF28'])] # GWANNv1 Mat AD within bottom 20 genes that passed significance
    # gdf.rename(columns={'symbol':'Gene', 'chrom':'Chrom'}, inplace=True)
    # gdf['P'] = 0

    # labels = {'MATERNAL_MARIONI':gdf}
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

    gpu_list = list(np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        run_genes.model_pipeline(exp_name='Sens1.4', label=label, 
                    param_folder=param_folder, gpu_list=gpu_list)
        # dummy_genes.model_pipeline(exp_name='Sens1.4Dummy', label=label, 
        #             param_folder=param_folder, gpu_list=gpu_list)
        break

def sensitivity_1_4_plots():
    agg_summ_df = pd.DataFrame(columns=['Gene', 'Acc', 'P', 'Grouping_id'])
    for i in [1,2,3,4]:
        summ_df = pd.read_csv(f'/home/upamanyu/GWANN/Code_AD/results_Sens1.4/{i}_MATERNAL_MARIONI_Sens1.4_summary.csv')
        summ_df['Grouping_id'] = i
        agg_summ_df = pd.concat((agg_summ_df, summ_df[agg_summ_df.columns]))
    
    sns.lineplot(data=agg_summ_df, x='Grouping_id', y='Acc', hue='Gene')
        
def sensitivity_1_5(chroms:list):
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'

    glist = (['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'] +
            ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
            ['ICAM3', 'ATP2C1', 'GLIS3', 'ARHGEF28']) # GWANNv1 Mat AD within bottom 20 genes that passed significance
    gpu_list = list(np.repeat([0, 1, 2, 3, 4], 4))
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        # run_genes.model_pipeline(exp_name='Sens1.5', label=label, 
        #             param_folder=param_folder, gpu_list=gpu_list,
        #             glist=glist)
        dummy_genes.model_pipeline(exp_name='Sens1.5Dummy', label=label, 
                    param_folder=param_folder, gpu_list=gpu_list)
        break

def sensitivity_1_6(chroms:list):
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'

    glist = (['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'] +
            ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
            ['ICAM3', 'ATP2C1', 'GLIS3', 'ARHGEF28']) # GWANNv1 Mat AD within bottom 20 genes that passed significance
    gpu_list = list(np.repeat([0, 1, 2, 3, 4], 4))
    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        # run_genes.model_pipeline(exp_name='Sens1.6', label=label, 
        #             param_folder=param_folder, gpu_list=gpu_list,
        #             glist=glist)
        dummy_genes.model_pipeline(exp_name='Sens1.6Dummy', label=label, 
                    param_folder=param_folder, gpu_list=gpu_list)
        break

def sensitivity_2():

    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens2'
    
    glist = (['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'] +
            ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
            ['ICAM3', 'ATP2C1', 'GLIS3', 'ARHGEF28']) # GWANNv1 Mat AD within bottom 20 genes that passed significance
    # glist = ['APOE']
    gpu_list = list(np.repeat([2, 3, 4, 5, 6, 7, 8, 9], 3))

    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
            sys_params = yaml.load(f, Loader=yaml.FullLoader)
            
        for grp_size in [20, 25, 30]:
            for si, seed in enumerate([82, 192, 8376, 1763]):
                exp_name = f'Sens2_{si}_{grp_size}'
                grp_id_path = f'{param_folder}/{exp_name}_group_ids_{label}.npz'
                
                create_groups(
                    label=label,
                    param_folder=param_folder, 
                    phen_cov_path='/mnt/sdg/UKB/Variables_UKB.txt',
                    grp_size=grp_size, oversample=grp_size,
                    random_seed=seed, grp_id_path=grp_id_path
                )

                # sys_params['GROUP_IDS_PATH'] = grp_id_path
                # sys_params['COV_ENC_PATH'] = f'{param_folder}/{exp_name}_cov_encodings_{label}.npz'

                # with open('{}/params_{}.yaml'.format(param_folder, label), 'r+') as f:
                #     yaml.dump(sys_params, f)
                    
                # cov_model.model_pipeline(label=label, param_folder=param_folder,
                #                          gpu_list=[5, 6], exp_suffix=exp_name, 
                #                          grp_size=grp_size)
                # cov_model.gen_cov_encodings(label=label, param_folder=param_folder,
                #                          device=0, exp_suffix=exp_name)
                
                # run_genes.model_pipeline(exp_name=exp_name, label=label, 
                #             param_folder=param_folder, gpu_list=gpu_list,
                #             glist=glist, grp_size=grp_size)
                # dummy_genes.model_pipeline(exp_name='Sens2Dummy', label=label, 
                #             param_folder=param_folder, gpu_list=gpu_list)
                break
            break
        break

def sensitivity_3():
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens3'
    
    glist = (['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'] +
            ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
            ['ICAM3', 'ATP2C1', 'GLIS3', 'ARHGEF28']) # GWANNv1 Mat AD within bottom 20 genes that passed significance
    # glist = ['APOE']
    gpu_list = list(np.repeat([2, 3, 4, 5, 6, 7, 8, 9], 3))

    for label in ['MATERNAL_MARIONI']:
        with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
            sys_params = yaml.load(f, Loader=yaml.FullLoader)
            
        for grp_size in [20, 25, 30]:
            for si, seed in enumerate([82, 192, 8376, 1763]):
                exp_name = f'Sens3_{si}_{grp_size}'
                grp_id_path = f'{param_folder}/{exp_name}_group_ids_{label}.npz'
                
                create_groups(
                    label=label,
                    param_folder=param_folder, 
                    phen_cov_path='/mnt/sdg/UKB/Variables_UKB.txt',
                    grp_size=grp_size, oversample=grp_size,
                    random_seed=seed, grp_id_path=grp_id_path
                )

                sys_params['GROUP_IDS_PATH'] = grp_id_path
                sys_params['COV_ENC_PATH'] = f'{param_folder}/{exp_name}_cov_encodings_{label}.npz'

                with open('{}/params_{}.yaml'.format(param_folder, label), 'r+') as f:
                    yaml.dump(sys_params, f)
                    
                # cov_model.model_pipeline(label=label, param_folder=param_folder,
                #                          gpu_list=gpu_list[:2], exp_suffix=exp_name, 
                #                          grp_size=grp_size)
                # cov_model.gen_cov_encodings(label=label, param_folder=param_folder,
                #                          device=gpu_list[0], exp_suffix=exp_name)
                
                run_genes.model_pipeline(exp_name=exp_name, label=label, 
                            param_folder=param_folder, gpu_list=gpu_list,
                            glist=glist, grp_size=grp_size)
                # dummy_genes.model_pipeline(exp_name='Sens2Dummy', label=label, 
                #             param_folder=param_folder, gpu_list=gpu_list)
            
def sensitivity_4():
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens4'
    
    glist = (['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'] +
            ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
            ['ICAM3', 'ATP2C1', 'GLIS3', 'ARHGEF28']) # GWANNv1 Mat AD within bottom 20 genes that passed significance
    # glist = ['APOE']
    gpu_list = list(np.repeat([1, 2, 3, 4, 5, 6, 7], 3))

    for label in ['MATERNAL_MARIONI']:
        with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
            sys_params = yaml.load(f, Loader=yaml.FullLoader)
            
        grp_size = 20
        for oversample in [10, 15]:
            for si, seed in enumerate([82, 192, 8376, 1763]):
                exp_name = f'Sens4_{si}_{oversample}'
                grp_id_path = f'{param_folder}/{exp_name}_group_ids_{label}.npz'
                
                create_groups(
                    label=label,
                    param_folder=param_folder, 
                    phen_cov_path='/mnt/sdg/UKB/Variables_UKB.txt',
                    grp_size=grp_size, oversample=oversample,
                    random_seed=seed, grp_id_path=grp_id_path
                )

                sys_params['GROUP_IDS_PATH'] = grp_id_path
                sys_params['COV_ENC_PATH'] = f'{param_folder}/{exp_name}_cov_encodings_{label}.npz'

                with open('{}/params_{}.yaml'.format(param_folder, label), 'w') as f:
                    yaml.dump(sys_params, f)
                    
                # cov_model.model_pipeline(label=label, param_folder=param_folder,
                #                          gpu_list=gpu_list[:2], exp_suffix=exp_name, 
                #                          grp_size=grp_size)
                # cov_model.gen_cov_encodings(label=label, param_folder=param_folder,
                #                          device=gpu_list[0], exp_suffix=exp_name)
                
                exp_name = f'Sens4.1_{si}_{oversample}'
                run_genes.model_pipeline(exp_name=exp_name, label=label, 
                            param_folder=param_folder, gpu_list=gpu_list,
                            glist=glist, grp_size=grp_size)
                # dummy_genes.model_pipeline(exp_name='Sens2Dummy', label=label, 
                #             param_folder=param_folder, gpu_list=gpu_list)
            
def sensitivity_5():
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens5'
    
    glist = (['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7', 'PPP1R37'] +
            ['ARSG', 'SMAD9', 'NFIA', 'SNRPB2'] + # GWANNv1 Mat AD within top 20 genes
            ['ICAM3', 'ATP2C1', 'GLIS3', 'ARHGEF28']) # GWANNv1 Mat AD within bottom 20 genes that passed significance
    # glist = ['APOE']
    gpu_list = list(np.repeat([1, 2, 3, 4, 5, 6, 7], 3))

    for label in ['MATERNAL_MARIONI']:
        with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
            sys_params = yaml.load(f, Loader=yaml.FullLoader)
            
        grp_size = 20
        for oversample in [15, 20]:
            for si, seed in enumerate([82, 192, 8376, 1763]):
                exp_name = f'Sens5_{si}_{oversample}'
                grp_id_path = f'{param_folder}/{exp_name}_group_ids_{label}.npz'
                
                # create_groups(
                #     label=label,
                #     param_folder=param_folder, 
                #     phen_cov_path='/mnt/sdg/UKB/Variables_UKB.txt',
                #     grp_size=grp_size, oversample=oversample,
                #     random_seed=seed, grp_id_path=grp_id_path
                # )

                sys_params['GROUP_IDS_PATH'] = grp_id_path
                sys_params['COV_ENC_PATH'] = f'{param_folder}/{exp_name}_cov_encodings_{label}.npz'
                sys_params['PARAMS_PATH'] = param_folder

                with open('{}/params_{}.yaml'.format(param_folder, label), 'w') as f:
                    yaml.dump(sys_params, f)
                    
                cov_model.model_pipeline(label=label, param_folder=param_folder,
                                         gpu_list=gpu_list[:2], exp_suffix=exp_name, 
                                         grp_size=grp_size)
                cov_model.gen_cov_encodings(label=label, param_folder=param_folder,
                                         device=gpu_list[0], exp_suffix=exp_name)
                
                run_genes.model_pipeline(exp_name=exp_name, label=label, 
                            param_folder=param_folder, gpu_list=gpu_list,
                            glist=glist, grp_size=grp_size)
                # dummy_genes.model_pipeline(exp_name='Sens2Dummy', label=label, 
                #             param_folder=param_folder, gpu_list=gpu_list)

def sensitivity_6():
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens6.1'
    
    with open('params/gene_subsets.yaml', 'r') as f:
        gdict = yaml.load(f, yaml.FullLoader)
    # glist = gdict['Marioni_meta'] + gdict['GWANN_v1_Meta']
    # glist = list(set(gdict['KEGG_AD'] + gdict['Marioni_meta']))
    glist = gdict['GWANN_v1_Meta']
    gdf = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv')
    gdf.set_index('symbol', inplace=True, drop=False)
    # gdf = gdf.loc[glist]
    gdf = gdf.loc[gdf.index.isin(glist)].drop_duplicates(subset=['symbol'])
    glist = gdf.index.to_list()
    
    # gpu_list = list(np.repeat([0, 1, 2, 3, 4, 5, 6, 7], 3))
    gpu_list = list(np.tile([5, 6, 7, 8, 9], 4))

    # Ceate data
    # for label in ['MATERNAL_MARIONI']:
        # for chrom, idx in gdf.groupby('chrom').groups.items():
        #     if str(chrom) not in chroms:
        #         continue
        #     gl = gdf.loc[idx]['symbol'].to_list()
        #     print(chrom, len(gl))
        #     print(gl)
        #     run_genes.create_csv_data(label=label, param_folder=param_folder, 
        #                               chrom=str(chrom), glist=gl, split=True)
        # run_genes.create_csv_data(label=label, param_folder=param_folder, 
        #                               chrom=str(1), glist=glist, split=True, 
        #                               num_procs=103)

    glist = [
        {'gene':'APOE', 'win':0, 'chrom':'19'},
        {'gene':'NDUFS2', 'win':0, 'chrom':'1'},
        {'gene':'PAX5', 'win':1, 'chrom':'9'},
        {'gene':'PAX5', 'win':4, 'chrom':'9'},
        # {'gene':'PAX5', 'win':6, 'chrom':'9'},
        {'gene':'BIN1', 'win':5, 'chrom':'2'}
    ]
    # glist = glist[-1:]

    for label in ['MATERNAL_MARIONI', 'PATERNAL_MARIONI']:
        with open('{}/params_{}.yaml'.format(param_folder, label), 'r') as f:
            sys_params = yaml.load(f, Loader=yaml.FullLoader)
            
        grp_size = 20
        for oversample in [20]:
            for si, seed in enumerate([192]):
                exp_name = f'Sens6.1'
                grp_id_path = f'{param_folder}/{exp_name}_group_ids_{label}.npz'
                
                create_groups(
                    label=label,
                    param_folder=param_folder, 
                    phen_cov_path='/mnt/sdg/UKB/Variables_UKB.txt',
                    grp_size=grp_size, oversample=oversample,
                    random_seed=seed, grp_id_path=grp_id_path
                )

                sys_params['GROUP_IDS_PATH'] = grp_id_path
                sys_params['COV_ENC_PATH'] = f'{param_folder}/{exp_name}_cov_encodings_{label}.npz'
                sys_params['PARAMS_PATH'] = param_folder

                with open('{}/params_{}.yaml'.format(param_folder, label), 'w') as f:
                    yaml.dump(sys_params, f)
                    
                # cov_model.create_cov_only_data(label=label, param_folder=param_folder)
                # cov_model.model_pipeline(label=label, param_folder=param_folder,
                #                          gpu_list=gpu_list[:2], exp_name=exp_name, 
                #                          grp_size=grp_size)
                # cov_model.gen_cov_encodings(label=label, param_folder=param_folder,
                #                          device=gpu_list[0], exp_name=exp_name)
                
                run_genes.model_pipeline(exp_name=exp_name, label=label, 
                            param_folder=param_folder, gpu_list=gpu_list,
                            glist=glist, grp_size=grp_size, shap_plots=True)
                
                # dummy_genes.create_dummy_pgen(param_folder=param_folder, label=label)
                # dummy_genes.model_pipeline(exp_name=f'{exp_name}Dummy', label=label, 
                #             param_folder=param_folder, gpu_list=gpu_list, 
                #             grp_size=grp_size)
        # break

def sensitivity_7():
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens7'
    
    with open('params/gene_subsets.yaml', 'r') as f:
        gdict = yaml.load(f, yaml.FullLoader)
    glist = list(set(gdict['KEGG_AD'] + gdict['Marioni_meta']))
    gdf = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv')
    gdf.set_index('symbol', inplace=True, drop=False)
    gdf = gdf.loc[gdf.index.isin(glist)].drop_duplicates(subset=['symbol'])
    gdf = gdf.loc[gdf['chrom'].isin([str(c) for c in range(2, 22, 2)])]
    print(gdf.chrom.unique())
    gdf.sort_index(inplace=True)
    glist = gdf.index.to_list()
    gpu_list = list(np.tile([5, 6, 7], 5))
    print(glist)
    
    for label in ['FH_AD']:
        for grp_size in [10]:
                torch_seed=int(os.environ['TORCH_SEED'])
                random_seed=int(os.environ['GROUP_SEED'])
                exp_name = f'Sens7_{torch_seed}{random_seed}_GS{grp_size}_v4'
                
                # cov_model.create_cov_only_data(label=label, param_folder=param_folder)
                # cov_model.model_pipeline(label=label, param_folder=param_folder,
                #                          gpu_list=gpu_list[:2], exp_name=exp_name, 
                #                          grp_size=grp_size)
                
                # run_genes.model_pipeline(exp_name=exp_name, label=label, 
                #             param_folder=param_folder, gpu_list=gpu_list,
                #             glist=glist, grp_size=grp_size, shap_plots=False)
                
                # dummy_genes.create_dummy_pgen(param_folder=param_folder, label=label)
                dummy_genes.model_pipeline(exp_name=f'{exp_name}Dummy', label=label, 
                            param_folder=param_folder, gpu_list=gpu_list, 
                            grp_size=grp_size)

def sensitivity_8():
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens8'
    
    with open('params/gene_subsets.yaml', 'r') as f:
        gdict = yaml.load(f, yaml.FullLoader)
    # glist = list(set(gdict['KEGG_AD'] + gdict['Marioni_meta']))
    # gdf = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv')
    # gdf.set_index('symbol', inplace=True, drop=False)
    # gdf = gdf.loc[gdf.index.isin(glist)].drop_duplicates(subset=['symbol'])
    # gdf = gdf.loc[gdf['chrom'].isin([str(c) for c in range(1, 22, 2)])]
    # print(gdf.chrom.unique())
    # gdf.sort_index(inplace=True)
    # glist = gdf.index.to_list()
    
    run1_df = pd.read_csv('/home/upamanyu/GWANN/Code_AD/results_Sens8_00_GS10_v4/FH_AD_Loss_Sens8_00_GS10_v4_gene_summary.csv')
    top200 = run1_df.sort_values(['P']).head(200)
    top200 = top200.loc[top200['Chrom'].isin([str(c) for c in range(1, 23, 2)])]
    glist = top200['Gene'].values

    gpu_list = list(np.tile([0, 1, 2, 3, 4], 5))

    for label in ['FH_AD']:
        for grp_size in [int(os.environ['GROUP_SIZE'])]:
                torch_seed=int(os.environ['TORCH_SEED'])
                random_seed=int(os.environ['GROUP_SEED'])
                exp_name = f'Sens8_{torch_seed}{random_seed}_GS{grp_size}_v8'
                
                cov_model.model_pipeline(label=label, param_folder=param_folder,
                                         gpu_list=gpu_list[:2], exp_name=exp_name, 
                                         grp_size=grp_size)
                
                run_genes.model_pipeline(exp_name=exp_name, label=label, 
                            param_folder=param_folder, gpu_list=gpu_list,
                            glist=glist, grp_size=grp_size, shap_plots=False)
                
                # dummy_genes.create_dummy_pgen(param_folder=param_folder, label=label)
                dummy_genes.model_pipeline(exp_name=f'Dummy{exp_name}', label=label, 
                            param_folder=param_folder, gpu_list=gpu_list, 
                            grp_size=grp_size)

if __name__ == '__main__':
    # sensitivity_1_2(chroms)
    # sensitivity_1_3(chroms)
    # sensitivity_1_4(chroms)
    # sensitivity_1_4_plots()
    # sensitivity_1_5(chroms)
    # sensitivity_1_6(chroms)
    # sensitivity_2()
    # sensitivity_3()
    # sensitivity_4()
    # sensitivity_5()
    sensitivity_8()