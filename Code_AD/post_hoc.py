import json
import os
import subprocess
import sys
from textwrap import wrap
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tqdm
from adjustText import adjust_text
from mygene import MyGeneInfo
from scipy.stats import skewnorm
from statsmodels.stats.multitest import multipletests

sys.path.append('..')
from GWANN.dataset_utils import get_win_snps


class EstimatePValue:
    def __init__(self, null_accs:np.ndarray) -> None:
        self.null_accs = null_accs
        self.moments = skewnorm.fit(null_accs)
    
    def plot_null_dist(self, plot_path:Optional[str]=None) -> None:
        
        plt.hist(self.null_accs, density=True, alpha=0.35)
        xs = [100, 200, 500, 800, 1000, 2000, 5000]
        ys = []
        for n in xs:
            np.random.seed(1047)    
            null_acc_sample = np.random.choice(self.null_accs, size=n)
            moments = skewnorm.fit(null_acc_sample)
            ys.append(moments)
            x = np.linspace(min(self.null_accs), max(self.null_accs), num=1000)
            y = skewnorm.pdf(x, *moments)
            plt.plot(x, y, linewidth=2, label=str(n))
        
        plt.legend(title='\n'.join(
            wrap('Distribution estimation sample size', 20)))
        plt.tight_layout()
        if plot_path is not None:
            plt.savefig(plot_path, dpi=100)
            plt.close()
        else:
            plt.show()

        # fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        # ax = ax.flatten()
        # for i in range(len(ax)):
        #     print([y[i] for y in ys])
        #     ax[i].plot(xs, [y[i] for y in ys])
        
        # plt.xlabel('Num dummy datasets')
        # plt.tight_layout()
        # plt.savefig('dist_moments.png', dpi=100)
        # plt.close()

    def estimate(self, acc:float) -> float:
        return skewnorm.sf(acc, *self.moments)

def calculate_p_values(label:str, exp_name:str, metric:str, greater_is_better:bool):
    if not os.path.exists(f'./results_{exp_name}'):
        os.mkdir(f'./results_{exp_name}')
    
    null_df = pd.read_csv(
        f'./NN_Logs/' + 
        f'{label}_ChrDummy{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
        f'{label}_ChrDummy{exp_name}_2500bp_summary.csv')
    if greater_is_better:
        ep = EstimatePValue(null_accs=null_df[metric].values)
    else:
        ep = EstimatePValue(null_accs=-1*null_df[metric].values)
    ep.plot_null_dist(f'./results_{exp_name}/{label}_{metric}_null_dist.png')

    summ_df = pd.read_csv(
        f'./NN_Logs/'+
        f'{label}_Chr{exp_name}_2500bp_summary.csv')
    if greater_is_better:
        summ_df['P'] = summ_df[metric].apply(lambda x: ep.estimate(x)).values
    else:
        summ_df['P'] = summ_df[metric].apply(lambda x: ep.estimate(-1*x)).values

    _, corr_p, _, alpha_bonf = multipletests(summ_df['P'].values, method='bonferroni')
    summ_df[f'P_bonf'] = corr_p
    summ_df[f'alpha_bonf'] = alpha_bonf
    summ_df[f'P_fdr_bh'] = multipletests(summ_df['P'].values, method='fdr_bh')[1]
    summ_df.to_csv(f'./results_{exp_name}/{label}_{metric}_{exp_name}_summary.csv', index=False)
    hits_df = summ_df.loc[summ_df['P_bonf'] < 0.05]
    hits_df.to_csv(f'./results_{exp_name}/{label}_{metric}_{exp_name}_hits.csv', index=False)

    gdf = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv')
    gdf.set_index('symbol', inplace=True)

    summ_df['Gene'] = summ_df['Gene'].apply(lambda x:x.split('_')[0]).values
    summ_df.sort_values(['Gene', 'P'], inplace=True)
    summ_df.drop_duplicates(['Gene'], inplace=True)
    summ_df['entrez_id'] = gdf.loc[summ_df['Gene'].values]['id'].values
    summ_df['ens_g'] = gdf.loc[summ_df['Gene'].values]['ens_g'].values
    summ_df.to_csv(f'./results_{exp_name}/{label}_{metric}_{exp_name}_gene_summary.csv', index=False)
    hits_df = summ_df.loc[summ_df['P_bonf'] < 0.05]
    hits_df.to_csv(f'./results_{exp_name}/{label}_{metric}_{exp_name}_gene_hits.csv', index=False)
    
def combine_chrom_summ_stats(chroms:list, label:str, exp_name:str):
    comb_summ_df = []
    for chrom in chroms:
        summ_df = pd.read_csv(
                    f'/mnt/sdb/NN_Logs/'+
                    f'Chr{chrom}_{label}_Chr{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
                    f'{label}_Chr{exp_name}_2500bp_summary.csv')
        comb_summ_df.append(summ_df)
    comb_summ_df = pd.concat(comb_summ_df)
    comb_summ_path = (f'./NN_Logs/' +
                        # f'{label}_Chr{exp_name}_GWANNet5_[32,16]_Dr_0.5_LR:0.005_BS:256_Optim:adam/'+
                        f'{label}_Chr{exp_name}_2500bp_summary.csv')
    if not os.path.exists(comb_summ_path):
        comb_summ_df.to_csv(comb_summ_path, index=False)
    else:
        print(f'Summary file exists at: {comb_summ_path}')

def combine_all_runs():
    seeds = [0, 4250, 937, 89, 172, 37, 363, 281, 142, 7282, 675, 893, 56, 
             265, 530, 3589]
    for dummy in ['Dummy', '']:
        merged = []
        for seed in seeds:
            df = pd.read_csv(f'./NN_Logs/' + 
                            f'FH_AD_Chr{dummy}Sens8_{seed}{seed}_GS10_v4_2500bp_summary.csv')
            df['Seed'] = seed
            df = df[['Gene', 'Seed', 'Chrom', 'SNPs', 'Epoch', 'Acc', 'Loss', 'ROC_AUC', 'Time']]
            print(f'{seed}: {np.sort(df["Chrom"].unique())}')
            merged.append(df)
        
        merged = pd.concat(merged).reset_index(drop=True)
        merged.sort_values(['Gene', 'Seed'], inplace=True, ignore_index=True)
        merged.drop_duplicates(['Gene', 'Seed'], inplace=True, ignore_index=True)
        if dummy == 'Dummy':
            merged.to_csv(f'./results_Sens8_v4/results_Sens8_dummy_combined.csv', 
                        index=False)
        else:
            merged.to_csv(f'./results_Sens8_v4/results_Sens8_combined.csv', 
                index=False)
        
        print(merged.shape)

def mine_agora(exp_name:str, hits_df_path:str) -> None:
    agora_res_path = f'./results_{exp_name}/enrichments/AGORA.csv'

    if not os.path.exists(agora_res_path):

        mg = MyGeneInfo()

        df = pd.read_csv(hits_df_path)
        df = df.loc[~df['pruned']]
        glist = df['Gene'].to_list()

        agora_df = pd.DataFrame(columns=['hgnc_symbol', 'ensembl_gene_id', 'is_igap', 'is_eqtl', 'target_nominations', 
            'is_any_rna_changed_in_ad_brain', 'is_any_protein_changed_in_ad_brain', 
            'DLPFC', 'TCX', 'CBE', 'STG', 'FP', 'PHG', 'IFG', 'BRAAK', 'CERAD', 'DCFDX', 'COGDX'])

        # for g in tqdm.tqdm(glist):
        for g in glist:
            ens_g = mg.query(f'symbol:{g.lower()}', species='human', 
                            fields='symbol,ensembl.gene')
            if 'hits' not in ens_g or len(ens_g['hits'])==0:
                print(f'[{g}]: No Ensembl gene id found')
                agora_df.loc[g, 'hgnc_symbol'] = g
                continue
            ens_g = ens_g['hits'][0]['ensembl']
            if isinstance(ens_g, list):
                ens_g = ens_g[0]['gene']
            else:
                ens_g = ens_g['gene']
            req_url = 'https://agora.adknowledgeportal.org/api/genes/{}'.format(ens_g)
            print(f'[{g}]: {req_url}')
            response = requests.get(req_url).json()
            
            for c in agora_df.columns:
                if c in response.keys():
                    if isinstance(response[c], list):
                        agora_df.loc[g, c] = len(response[c])
                    else:
                        agora_df.loc[g, c] = response[c]

            neuro_corr = response['neuropathologic_correlations']
            AD_hallmarks = {n['neuropath_type']:n['pval_adj'] for n in neuro_corr}
            
            for h, v in AD_hallmarks.items():
                agora_df.loc[g, h] = v

        agora_df = agora_df.fillna('NA')
        agora_df.to_csv(agora_res_path, index=False)
    else:
        agora_df = pd.read_csv(agora_res_path)

    # Agora heatmap
    agora_df = agora_df[['is_igap', 'is_any_rna_changed_in_ad_brain', 
                         'is_any_protein_changed_in_ad_brain', 'is_eqtl', 
                         'BRAAK', 'CERAD', 'COGDX', 'hgnc_symbol']]
    agora_df['COGDX'] = agora_df['COGDX'] < 0.05
    agora_df['BRAAK'] = agora_df['BRAAK'] < 0.05
    agora_df['CERAD'] = agora_df['CERAD'] < 0.05
    agora_df.set_index('hgnc_symbol', inplace=True)
    agora_df = agora_df.apply(lambda x:x.astype(float), axis=0)
    agora_df.sort_values(agora_df.columns.to_list(), ascending=False, inplace=True)
    agora_df = agora_df.T
    
    sns.set(font_scale=0.9)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.heatmap(data=agora_df, cmap='Reds', cbar=False, xticklabels=True, 
                linewidths=0.2, linecolor='gray', ax=ax)
    ax.set_xlabel('')
    fig.tight_layout()
    fig.savefig(f'./results_{exp_name}/enrichments/AGORA_heatmap.svg')
    fig.savefig(f'./results_{exp_name}/enrichments/AGORA_heatmap.png', dpi=100)
    plt.close()

def manhattan(exp_name:str, exp_summary_file:str, hits_df_path:str, 
              p_thresh:float, p_nominal_thresh:float, p_col:str='P') -> None:
    
    summ_df = pd.read_csv(exp_summary_file)
    summ_df = summ_df.loc[~summ_df['Chrom'].isna()]
    summ_df['Chrom'] = summ_df['Chrom'].astype(int).values
    summ_df['Win'] = summ_df['Gene'].apply(lambda x:int(x.split('_')[1])).values
    summ_df['Gene'] = summ_df['Gene'].apply(lambda x:x.split('_')[0]).values
    non_0_min_P = summ_df.loc[summ_df[p_col] > 0][p_col].min()
    summ_df.loc[summ_df[p_col] == 0, p_col] = non_0_min_P/10
    print(f'Set all 0 P values to {non_0_min_P/10}')

    summ_df.sort_values(['Gene', 'stat_trial_A'], 
                        ascending=[True, True], 
                        inplace=True)
    summ_df.drop_duplicates(['Gene'], inplace=True)

    hits_df = pd.read_csv(hits_df_path)
    hits = hits_df.loc[~hits_df['pruned']]['Gene'].to_list()

    genes_df = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv',
                           dtype={'chrom':str})
    genes_df.set_index('symbol', inplace=True)
    plt.figure(figsize=(10,7))

    xs = []
    data = []
    lines = []
    chrom_size_dict = {}
    prev = 0
    ticks = np.arange(1, 23)
    for c in range(1, 23):
        temp = [0,0] 
        temp[0] = prev
        temp[1] = prev + (np.max(genes_df.loc[genes_df['chrom'] == str(c)]['end']))/(10**7)
        prev = temp[1]
        chrom_size_dict[c] = temp
        ticks[c-1] = np.mean(temp)

    colors = np.tile(['mediumblue', 'deepskyblue'], 11)
    markers = np.repeat(['o'], 22)
    texts = []
    i = 0
    for chrom, idxs in summ_df.groupby('Chrom').groups.items():
        df = summ_df.loc[idxs]
        pos = genes_df.loc[df['Gene']]['start'].values
        pos = chrom_size_dict[chrom][0] + pos/(10**7)        
        x = pos
        xs.append(x)
        data.append(-np.log10(df[p_col].values))
        lines.append(plt.scatter(xs[-1], data[-1], alpha=0.5, s=5, 
            marker=markers[i], color=colors[i]))
        for ind in range(len(xs[-1])):
            if df.iloc[ind]['Gene'] in hits:
                texts.append(plt.text(xs[-1][ind], data[-1][ind], df.iloc[ind]['Gene'], 
                    fontdict={'size':6}, rotation=90))
        
        i += 1
    
    adjust_text(texts, force_text=(0.4, 0.5), force_points=(0.5, 0.8),
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    plt.axhline(-np.log10(p_nominal_thresh), linestyle='--', alpha=0.5, color='k', linewidth=0.5)
    plt.axhline(-np.log10(p_thresh), linestyle='--', alpha=0.5, color='r', linewidth=0.5)
    plt.yticks(fontsize=14)
    plt.xticks(ticks, np.arange(1, 23), fontsize=14, rotation=90)
    plt.grid(axis='x', alpha=0.3)
    plt.xlabel('Chrom', fontsize=14)
    plt.ylabel('-log10 (P)', fontsize=14)
    plt.title('{} - manhattan'.format(exp_name))
    
    fname = f'./results_{exp_name}/enrichments/manhattan'
    plt.tight_layout()
    plt.savefig(f'{fname}.svg', bbox_inches='tight')
    plt.savefig(f'{fname}.png', dpi=100)
    plt.close()

def ld_link_matrix(snp_list:list, out_file:str):
    cmd = 'bash ./post_hoc_scripts/ld_link_matrix.sh '
    cmd += f'{out_file} '
    snps = r"\\n".join(snp_list)
    cmd += f'{snps}'
    subprocess.run(cmd, shell=True)

def hit_gene_LD(exp_name:str, exp_summary_file:str, p_thresh:float, 
                pgen_data_base:str) -> None:
    samples_df = pd.read_csv('/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun_Sens8/all_ids_FH_AD.csv')
    samples_df = samples_df[['iid']]
    samples_df['#FID'] = samples_df['iid'].values
    samples_df.rename(columns={'iid':'IID'}, inplace=True)
    samples_df[['#FID', 'IID']].to_csv(f'./results_{exp_name}/LD/keep.csv', index=False, sep='\t')
    
    summ_df = pd.read_csv(exp_summary_file)
    summ_df = summ_df.loc[~summ_df['Chrom'].isna()]
    summ_df['Chrom'] = summ_df['Chrom'].astype(int).values
    summ_df['Win'] = summ_df['Gene'].apply(lambda x:int(x.split('_')[1])).values
    summ_df['Gene'] = summ_df['Gene'].apply(lambda x:x.split('_')[0]).values
    hit_df = summ_df.loc[summ_df['p_stat_trial_A'] < p_thresh].copy()
    
    genes_df = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv',
                           dtype={'chrom':str})
    genes_df.set_index('symbol', inplace=True)

    hit_df['start'] = genes_df.loc[hit_df['Gene'].values]['start'].values - 2500
    hit_df['end'] = genes_df.loc[hit_df['Gene'].values]['end'].values + 2500
    hit_df.sort_values(['Chrom', 'Gene', 'Win'], inplace=True)

    # Retrieve SNPs of hit windows
    snps = []
    genes = []
    for chrom, idxs in tqdm.tqdm(hit_df.groupby('Chrom').groups.items()):
        cdf = hit_df.loc[idxs]
        pgen_data = f'{pgen_data_base}/UKB_chr{chrom}.pvar'
        if not os.path.exists(pgen_data):
            print(f'{pgen_data} not on this server' )
            continue
        for _, r in cdf.iterrows():
            snp_df = get_win_snps(chrom=str(chrom), start=r['start'], end=r['end'], 
                         win=r['Win'], pgen_data=pgen_data)
            snps.append(snp_df)
            genes.extend([r['Gene']]*len(snp_df))

    snp_df = pd.concat(snps)
    snp_df['GENE'] = genes
    snp_df.rename(columns={'CHROM':'#CHROM'}, inplace=True)
    snp_df.to_csv(f'./results_{exp_name}/LD/hit_snps.csv', index=False, sep='\t')

    # Perform LD calculations between hit snps
    gene_pair_ld = []
    for chrom in snp_df['#CHROM'].unique():
        chrom_df = snp_df.loc[snp_df['#CHROM']==chrom].copy()
        chrom_df.drop_duplicates(['ID'], inplace=True)
        
        snp_list = chrom_df['ID'].to_list()
        out_file = f'./results_{exp_name}/LD/chrom{chrom}_LDMatrix.csv'
        ld_link_matrix(snp_list=snp_list, out_file=out_file)

        chrom_df.set_index('ID', inplace=True)
        ld_mat = pd.read_csv(out_file, sep='\t', index_col=0)
        ld_list = ld_mat.stack().reset_index()
        ld_list.columns = ['SNP1', 'SNP2', 'r2']
        ld_list['Gene1'] = chrom_df.loc[ld_list['SNP1']]['GENE'].values
        ld_list['Gene2'] = chrom_df.loc[ld_list['SNP2']]['GENE'].values
        ld_list = ld_list.groupby(['Gene1', 'Gene2'])['r2'].max().reset_index()
        gpair = ld_list[['Gene1', 'Gene2']].apply(sorted, axis=1).values
        ld_list['Gene1'] = [g[0] for g in gpair]
        ld_list['Gene2'] = [g[1] for g in gpair]
        ld_list['Chrom'] = chrom
        ld_list.drop_duplicates(inplace=True)
        ld_list = ld_list.loc[ld_list['Gene1']!=ld_list['Gene2']]
        ld_list = ld_list.loc[ld_list['r2'] >= 0.8]
        gene_pair_ld.append(ld_list)

        os.remove(out_file)

    gene_pair_ld = pd.concat(gene_pair_ld)
    gene_pair_ld.to_csv(f'./results_{exp_name}/LD/gene_pair_LD.csv', index=False)

def hit_LD_prune(exp_name:str, exp_summary_file:str, p_thresh:float) -> None:
    ld_pair = pd.read_csv(f'./results_{exp_name}/LD/gene_pair_LD.csv')

    summ_df = pd.read_csv(exp_summary_file)
    summ_df = summ_df.loc[~summ_df['Chrom'].isna()]
    summ_df['Chrom'] = summ_df['Chrom'].astype(int).values
    summ_df['Win'] = summ_df['Gene'].apply(lambda x:int(x.split('_')[1])).values
    summ_df['Gene'] = summ_df['Gene'].apply(lambda x:x.split('_')[0]).values
    hit_df = summ_df.loc[summ_df['p_stat_trial_A'] < p_thresh].copy()
    hit_df.sort_values(['Gene', 'stat_trial_A'], ascending=[True, True], 
                        inplace=True)
    hit_df.drop_duplicates(['Gene'], inplace=True)

    hit_df['pruned'] = False
    for g1, g2 in ld_pair[['Gene1', 'Gene2']].values:
        # Prune gene in the pair with higher statistic value (loss)
        try:
            pair_stats = hit_df.loc[
                hit_df['Gene'].isin([g1, g2])]
            if set(pair_stats['Gene']) != set([g1, g2]):
                continue
            pruned_index = pair_stats['stat_trial_A'].sort_values().index[-1]
            hit_df.loc[pruned_index, 'pruned'] = True
        except IndexError:
            print(f'{g1} or {g2} not in hit_df.')
            
    hit_df.to_csv(f'./results_{exp_name}/LD/pruned_gene_hits_{p_thresh:.0e}.csv', index=False)

if __name__ == '__main__':
    # for rseed in [937, 89, 172, 37, 363, 142]:
    #     label = 'FH_AD'
    #     exp_name = f'Sens8_{rseed}{rseed}_GS10_v4'
    #     print(exp_name)
    #     combine_chrom_summ_stats(list(range(1, 23, 1)), 
    #                             label=label, exp_name=exp_name)
    # combine_all_runs()
    
    # FDR 0.05 p-thresh = 1e-23
    # FDR 0.1 p-thresh = 1e-14
    # hit_gene_LD(exp_name='Sens8_v4', 
    #             exp_summary_file='./results_Sens8_v4/summary_nsuperwins_1.csv', 
    #             p_thresh=1e-14, 
    #             pgen_data_base='/mnt/sdh/upamanyu/GWANN_pgen')
    # for p_thresh in [1e-23, 1e-14]:
    #     hit_LD_prune(exp_name='Sens8_v4', 
    #                 exp_summary_file='./results_Sens8_v4/summary_nsuperwins_1.csv', 
    #                 p_thresh=p_thresh)
    
    # manhattan(exp_name='Sens8_v4', 
    #           exp_summary_file='./results_Sens8_v4/summary_nsuperwins_1.csv', 
    #           hits_df_path='./results_Sens8_v4/LD/pruned_gene_hits_1e-23.csv', 
    #           p_thresh=1e-23, p_nominal_thresh=1e-14, p_col='p_stat_trial_A')
    mine_agora(exp_name='Sens8_v4', 
               hits_df_path='./results_Sens8_v4/LD/pruned_gene_hits_1e-23.csv')