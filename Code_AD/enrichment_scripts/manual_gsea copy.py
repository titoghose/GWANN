import math
import multiprocessing as mp
import os
from functools import partial
from itertools import islice
import networkx as nx

import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import yaml
from scipy.stats import binom_test, ks_2samp, norm, hypsecant
from tqdm import tqdm

tqdm.pandas()
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'

from bs4 import BeautifulSoup
import requests
import seaborn as sns
from upsetplot import UpSet, from_contents, plot

perm_ranks = dict()
PHENO = 'AD'

def map_genes_by_pos(gdf, col, x):
    pos = x['gene_pos']
    chrom = x['CHR']

    glist = gdf.loc[
                gdf['chrom'] == chrom].loc[
                    (pos >= gdf['start_hg38']) & (pos <= gdf['end_hg38'])][col].to_list()
    if col == 'entrez_id':
        return ';'.join([str(int(g)) for g in glist])
    else:
        return ';'.join([str(g) for g in glist])

def gene_remap(res_path):
    res_df = pd.read_csv(res_path)
    res_df = res_df.astype({'CHR':str})
    
    gdf = pd.read_csv('/home/upamanyu/GWASOnSteroids/GWASNN/datatables/gene_list2.csv')
    gdf = gdf.astype({'chrom':str})
    gdf.dropna(subset=['start_hg38', 'end_hg38'], inplace=True)
    gdf.dropna(subset=['entrez_id'], inplace=True)
    gdf.dropna(subset=['symbol'], inplace=True)
    
    # print(sorted(res_df.progress_apply(partial(map_genes_by_pos, gdf, 'symbol'), axis=1).values)[-20:])
    res_df['new_entrez_id'] = res_df.progress_apply(partial(map_genes_by_pos, gdf, 'entrez_id'), axis=1).values
    res_df['new_gene'] = res_df.progress_apply(partial(map_genes_by_pos, gdf, 'symbol'), axis=1).values
    

    res_df.to_csv(res_path, index=False)

def remove_genes_from_path(glist, pdict):
    new_pdict = pdict
    for p, gs in pdict.items():
        new_gs = gs
        for g in gs:
            if g not in glist:
                new_gs.remove(g)
        if len(new_gs) == 0:
            continue
        else:
            new_pdict[p] = new_gs
    
    return new_pdict

def get_chunks(data, SIZE=10000):
        it = iter(data)
        for i in range(0, len(data), SIZE):
            yield {k:data[k] for k in islice(it, SIZE)}

def get_pathway_dict():
    with open('filtered_pathways.yaml', 'r') as f:
        pathway_dict = yaml.load(f, yaml.FullLoader)

    return pathway_dict

def filter_pathway_dict():

    with open('/mnt/sdb/Pathway_data/c2.all.v7.5.1.entrez.yaml', 'r') as f:
        pathway_dict = yaml.load(f, yaml.FullLoader)
    pathway_dict = {k:v for k,v in pathway_dict.items() if (k.startswith('KEGG') or k.startswith('WP') or k.startswith('REACTOME'))}
    print(len(pathway_dict))

    gpath = '/home/upamanyu/GWASOnSteroids/GWASNN/datatables/gene_list2.csv'
    gdf = pd.read_csv(gpath)
    gdf = gdf.loc[~gdf['entrez_id'].isnull()]
    gdf = gdf.loc[gdf['chrom'].isin([str(c) for c in range(1, 23)])]
    glist = set([str(int(eid)) for eid in gdf['entrez_id'].to_list()])
    print(gdf.shape)

    new_pathway_dict = {}
    for pw, pwgs in pathway_dict.items():
        inter = set(pwgs).intersection(glist)
        if len(inter) == 0:
            print(pw)
            continue
        new_pathway_dict.update({pw:list(inter)})

    with open('filtered_pathways.yaml', 'w') as f:
        yaml.dump(new_pathway_dict, f)

def get_glist(df, col):
    gs = df[col].to_list()
    glist = []
    for g in gs:
        if isinstance(g, float) and (math.isnan(g)):
            continue 
        if isinstance(g, str) and (g=='nan' or g==''):
            continue
        else:
            tmp = g.split(';')
            glist.extend(tmp)
    
    if isinstance(glist[0], float) or isinstance(glist[0], int):
        glist = [str(int(g)) for g in glist]
    else:
        glist = [str(g) for g in glist]
    glist = list(set(glist))
    
    return glist

def get_eid(x):
    if isinstance(x, float) and (math.isnan(x)):
        return ''
    if isinstance(x, str) and (x=='nan' or x==''):
        return ''
    
    gl = x.split(';')
    if len(gl) == 0:
        return ''
    else:
        if isinstance(gl[0], float) or isinstance(gl[0], int):
            return str(int(gl[0]))
        else:
            return str(gl[0])

def parallel_pw_enrich(ranked_gdf, alpha, col, pathway_dict):
    pw_p_dict = {}
    n_pws = 0
    perm_scores = []
    for pw, v in tqdm(pathway_dict.items()):
        tdf = ranked_gdf.loc[ranked_gdf[col].isin(v)]
        print(pw)

        # If at least 50% of genes in the GO geneset have not been
        # analysed in the GWAS, drop the GO term        
        if len(tdf)/len(v) < 0.5:
            print('{:<60}{}/{}'.format(pw, len(tdf), len(v)))
        
        if len(tdf) == 0:
            continue
        
        # else:
        G = tdf[col].to_list()
        nk = len(G)
        S1_prob_dist = global_rank(G, ranked_gdf.values, alpha=alpha)
        
        G_ = list(set(ranked_gdf[col].to_list()).difference(set(G)))
        S2_prob_dist = global_rank(G_, ranked_gdf.values, alpha=0)
        
        assert np.allclose(np.cumsum(S1_prob_dist)[-1:], [np.sum(S1_prob_dist),])
        assert np.allclose(np.cumsum(S2_prob_dist)[-1:], [np.sum(S2_prob_dist),])

        S1_F = np.cumsum(S1_prob_dist)/np.sum(S1_prob_dist)
        S2_F = np.cumsum(S2_prob_dist)/np.sum(S2_prob_dist)

        assert len(
            set(np.where(S1_prob_dist == 0)[0]).intersection(
            set(np.where(S2_prob_dist == 0)[0]))) == 0

        # plt.step(np.arange(len(S1_F)), S1_F, color='blue', label='pw_CDF')
        # plt.step(np.arange(len(S2_F)), S2_F, color='orange', label='nonpw_CDF')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig('{}.png'.format(pw))

        S = np.max(S1_F - S2_F)
        S_pis = get_random_rank_dist(G, ranked_gdf.values, alpha)

        S_norm = S / np.mean(S_pis)
        S_pis = S_pis / np.mean(S_pis)
        perm_scores.extend(S_pis)

        pk = (1/(len(S_pis)+1))*(np.count_nonzero(S_pis >= S_norm)+1)

        pw_p_dict[pw] = (pk, S_norm, tdf[col].to_list(), len(v))
        n_pws += 1
        # break
    
    return pw_p_dict, perm_scores

def parallel_GO_enrich(ranked_gdf, alpha, go_df, col, go_dict):
    go_p_dict = {}
    n_pws = 0
    perm_scores = []
    for go, idxs in tqdm(go_dict.items()):
        go_genes = list(go_df.loc[idxs]['symbol'].unique())
        tdf = ranked_gdf.loc[ranked_gdf[col].isin(go_genes)]

        # If at least 80% of genes in the GO geneset have not been
        # analysed in the GWAS, drop the GO term
        if len(tdf)/len(go_genes) < 0.5:
            continue
        else:
            G = tdf[col].to_list()
            nk = len(G)
            S1_prob_dist = global_rank(G, ranked_gdf.values, alpha)
            # tmp = global_rank_(G, ranked_gdf.values, alpha=alpha)
            # assert np.array_equal(tmp, S1_prob_dist)
            
            G_ = list(set(ranked_gdf[col].to_list()).difference(set(G)))
            S2_prob_dist = global_rank(G_, ranked_gdf.values, alpha=0)

            assert np.cumsum(S1_prob_dist)[-1] == 1
            assert np.cumsum(S2_prob_dist)[-1] == 1
            
            S = np.max(np.cumsum(S1_prob_dist) - np.cumsum(S2_prob_dist))
            S_pis = get_random_rank_dist(G, alpha)

            S_norm = S / np.mean(S_pis)
            S_pis = S_pis / np.mean(S_pis)
            perm_scores.extend(S_pis)

            pk = (1/(len(S_pis)+1))*(np.count_nonzero(S_pis >= S_norm)+1)

            go_p_dict[go] = (pk, S_norm, tdf[col].to_list())
            n_pws += 1
    
    return go_p_dict, perm_scores

def GSEA_pathway_enrichment(res_path, enr_res_path, pw_subset, pheno, rank,
                                col='entrez_id', alpha=0):  

    global perm_ranks
    
    ranked_gdf = local_rank(res_path, col, rank)
    
    # Background genes
    glist = list(ranked_gdf[col].unique())
    # glist = get_glist(ranked_gdf, col)
    pathway_dict = get_pathway_dict()

    for pw_set in pw_subset:
        if os.path.isfile(enr_res_path):
            continue
        print(pw_set,'\n')
        pw_dict = {}
        n_pws = 0
        for pw, v in tqdm(pathway_dict.items()):
            if pw.startswith(pw_set):
                pw_dict.update({pw:v})
                n_pws += 1
                
        n_procs = 60
        if n_pws < n_procs:
            n_procs = n_pws
        fargs = get_chunks(pw_dict, int(np.floor(n_pws/n_procs)))
            
        pw_p_dict = {}
        perm_scores = []
        cover = []
        with mp.Pool(n_procs) as pool:
            res = pool.map(partial(parallel_pw_enrich, ranked_gdf, 
                alpha, col), fargs)
            for r in res:
                pw_p_dict.update(r[0])
                perm_scores.extend(r[1])

        sig_pws = sorted(list(pw_p_dict.keys()), key=lambda x:pw_p_dict[x][0])
        res = 'p\tNES\tcoverage\tpathway\n'
        for spw in sig_pws:
            res += '{:e}\t{}\t{}/{}\t{}\n'.format(
                pw_p_dict[spw][0], pw_p_dict[spw][1], 
                len(pw_p_dict[spw][2]), pw_p_dict[spw][3], spw)

        with open(enr_res_path, 'w') as f:
            f.write(res)
        df = pd.read_csv(enr_res_path, sep='\t')
        df.sort_values(['NES', 'p'], ascending=[False, True], inplace=True)
        df.to_csv(enr_res_path, sep='\t', index=False)

        np.save(enr_res_path.replace('.txt', '.npy'), np.array(perm_scores))

def GSEA_GO_enrichment(res_path, pheno, rank, go_sub_name,
                        col='symbol', go_subset=None, alpha=0):
    """https://www.pathwaycommons.org/guide/primers/data_analysis/gsea/

    Parameters
    ----------
    res_path : [type]
        [description]
    go_subset : [type], optional
        [description], by default None
    """
    global perm_ranks

    enr_res_path = '../manual_enrich_{}_rank_{}/'\
            '{}_GSEA_{}_enrich.txt'.format(pheno, rank, pheno, go_sub_name)
    if os.path.isfile(enr_res_path):
        return

    ranked_gdf = local_rank(res_path, col, rank)
    perm_ranks['target'] = perm_ranks[pheno]
    perm_ranks['target'].set_index(col, drop=False, inplace=True)
    perm_ranks['target'] = perm_ranks['target'].loc[ranked_gdf[col].values]
    perm_ranks['target'] = perm_ranks['target'][[col,]+[str(i) for i in range(1000)]]
    
    go_df = pd.read_csv('/home/upamanyu/Will_analysis/data/GO_data/goa_human.gaf', 
        sep='\t', comment='!', header=None)
    go_df = go_df.iloc[:, [1, 2, 3, 4]]
    go_df.columns = ['uniprot_id', 'symbol', 'qualifier', 'go_id']
    go_df.drop_duplicates(['symbol', 'go_id'], inplace=True)
    go_df.drop_duplicates(['uniprot_id', 'go_id'], inplace=True)
    go_map = pd.read_csv('/home/upamanyu/Will_analysis/data/GO_data/go_mapping.csv')
    go_map.set_index('go_id', inplace=True, drop=False)
    go_df['go_name'] = go_map.loc[go_df['go_id'].values]['go_name'].values
    if go_subset is not None:
        go_df = go_df.loc[go_df['go_id'].isin(go_subset)]
    go_grps = go_df.groupby('go_id').groups

    n_go = 0
    go_dict = {}
    for go, idxs in tqdm(go_grps.items()):
        go_genes = list(go_df.loc[idxs]['symbol'].unique())
        tdf = ranked_gdf.loc[ranked_gdf[col].isin(go_genes)]

        # If at least 80% of genes in the GO geneset have not been
        # analysed in the GWAS, drop the GO term
        if len(tdf)/len(go_genes) < 0.5:
            continue
        else:
            go_dict.update({go:idxs})
            n_go += 1
    
    n_procs = 60
    if n_go < n_procs:
        n_procs = n_go
    fargs = get_chunks(go_dict, int(np.floor(n_go/n_procs)))
    go_p_dict = {}
    perm_scores = []
    
    with mp.Pool(n_procs) as pool:
        res = pool.map(partial(parallel_GO_enrich, ranked_gdf, 
            alpha, go_df, col), fargs)
        for r in res:
            go_p_dict.update(r[0])
            perm_scores.extend(r[1])

    sig_gos = sorted(list(go_p_dict.keys()), key=lambda x:go_p_dict[x][0])
    res = 'p\tNES\tGO_id\tGO_name\tnum_genes\n'
    for sgo in sig_gos:
        res += '{:e}\t{}\t{}\t{}\t{}\n'.format(
            go_p_dict[sgo][0], go_p_dict[sgo][1], sgo, 
            go_map.loc[sgo]['go_name'], len(go_p_dict[sgo][2]))
    
    with open(enr_res_path, 'w') as f:
        f.write(res)
    df = pd.read_csv(enr_res_path, sep='\t')
    df.sort_values(['NES', 'p'], ascending=[False, True], inplace=True)
    df.to_csv(enr_res_path, sep='\t', index=False)
        
    np.save(enr_res_path.replace('.txt', '.npy'), np.array(perm_scores))

def local_rank(res_path, col, rank='logP'):
    res_df = pd.read_csv(res_path)
    res_df = res_df.loc[res_df['P'] != 0]

    # old_hits = ["EXOC4", "NRXN1", "NRXN3", "PAX5", "CADM2", "RBMS3", "LUZP2", 
    #         "IMM2PL", "MACROD2", "EFNA5", "EPHA1", "APOE", "APOC1", "TOMM40", 
    #         "EXOC3L2", "BCAM", "PPP1R37", "GEMIN7", "APH1B", "WWOX", "GLIS3", 
    #         "ROBO1", "BDNF"]
    # res_df = res_df.loc[~res_df['Gene'].isin(old_hits)]

    # if 'entrez_id' in col and str(res_df[col].dtypes).startswith('float'):
    #     res_df[col] = res_df[col].apply(lambda x: str(int(x)) if not math.isnan(x) else '').values
    # res_df[col] = res_df[col].apply(get_eid).values
    res_df[col] = res_df[col].apply(lambda x: str(int(x)) if not math.isnan(x) else '').values
    res_df.sort_values(['P', col], inplace=True)
    res_df.drop_duplicates([col], inplace=True)

    if rank == 'logP':
        res_df['rank'] = -np.log10(res_df['P'].values)
    elif rank == 'Acc':
        res_df['rank'] = res_df['Acc'].values
    elif rank == 'logPxAcc':
        res_df['rank'] = -np.log10(res_df['P'].values) * res_df['Acc'].values
    elif rank == 'noAPOE':
        res_df['rank'] = -np.log10(res_df['P'].values) * res_df['Acc'].values
        res_df = res_df.loc[~(res_df['symbol'].isin(
            ['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7']))]
    elif rank == 'noAPOE_Acc':
        res_df['rank'] = res_df['Acc'].values
        res_df = res_df.loc[~(res_df['symbol'].isin(
            ['APOE', 'APOC1', 'TOMM40', 'BCAM', 'GEMIN7']))]
    
    res_df.sort_values('rank', ascending=False, inplace=True)
    
    return res_df[[col, 'rank']]

def get_random_rank_dist(G, ranked_glist, alpha):
    S_pis = []
    genes = list(ranked_glist[:, 0])
    r = ranked_glist[:, 1]
    np.random.seed(7356)
    seeds = np.random.choice(np.arange(1, 10089), 1000, replace=False)
    for j in range(1, 1000):
        
        np.random.seed(seeds[j])
        G_pi = np.random.choice(genes, len(G), replace=False)
        S1_prob_dist = global_rank(G_pi, ranked_glist, alpha=alpha)
        
        G_ = list(set(genes).difference(set(G_pi)))
        S2_prob_dist = global_rank(G_, ranked_glist, alpha=0)
        assert len(S1_prob_dist) == len(S2_prob_dist)
        
        # S_pi, _ = ks_2samp(S1_prob_dist, S2_prob_dist, alternative='lower')
        S1_F = np.cumsum(S1_prob_dist)/np.sum(S1_prob_dist)
        S2_F = np.cumsum(S2_prob_dist)/np.sum(S2_prob_dist)

        S_pi = np.max(S1_F - S2_F)
        S_pis.append(S_pi)

    S_pis = np.array(S_pis)

    return S_pis

def get_random_rank_dist_pheno_perm(G, alpha):
    global perm_ranks

    S_pis = []
    genes = perm_ranks['target'].iloc[:, 0].to_list()
    for j in range(1, 1000):
        null_ranked_glist = perm_ranks['target'].iloc[:, [0, j+1]].values
        # print(null_ranked_glist)
        
        S1_prob_dist = global_rank(G, null_ranked_glist, alpha=alpha)
        
        G_ = list(set(genes).difference(set(G)))
        S2_prob_dist = global_rank(G_, null_ranked_glist, alpha=0)
        assert len(S1_prob_dist) == len(S2_prob_dist)
        
        # S_pi, _ = ks_2samp(S1_prob_dist, S2_prob_dist, alternative='lower')
        S1_F = np.cumsum(S1_prob_dist)/np.sum(S1_prob_dist)
        S2_F = np.cumsum(S2_prob_dist)/np.sum(S2_prob_dist)

        S_pi = np.max(S1_F - S2_F)
        S_pis.append(S_pi)

    S_pis = np.array(S_pis)

    return S_pis

def global_rank(gset, ranked_glist, alpha):
    S_prob_dist = np.zeros(len(ranked_glist))
    denom = 0
    g = list(ranked_glist[:, 0])
    r = ranked_glist[:, 1]
    # g_idx = [g.index(gs) for gs in gset]
    g_idx = np.where(np.in1d(g, gset))[0]
    S_prob_dist[g_idx] = r[g_idx]**alpha
    denom = np.sum(S_prob_dist)
    # S_prob_dist /= denom

    return S_prob_dist

def pw_enrichment_hist(enrichment_tables, titles, base, topk=10):

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(len(enrichment_tables)):
        enrichment_table = enrichment_tables[i]
        ax = axes[i]
        title = titles[i]

        idxs = np.flip(np.argsort(enrichment_table[:, 2]))
        enrichment_table = enrichment_table[idxs, :]
        x = enrichment_table[:topk, 3]
        heights = np.array(enrichment_table[:topk, 2], dtype=float)
        
        ax.bar(x, heights, linewidth=0)
        ax.set_xticklabels(x, rotation=90, fontsize=6)
        ax.set_title(title)
        # ax.set_ylabel('Uncorrected -log10(P)')
        ax.set_ylabel('Enrichment score')
        # ax.set_xlabel('Pathway/GO term')
    
    fig.tight_layout()
    fig.savefig('{}/PW_enrich.png'.format(base))

def GO_enrichment_hist(enrichment_table, title, base, topk=10):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    idxs = np.flip(np.argsort(enrichment_table[:, 2]))
    enrichment_table = enrichment_table[idxs, :]
    x = enrichment_table[:topk, 3]
    # heights = -np.log10(np.array(enrichment_table[:topk, ], dtype=float))
    heights = np.array(enrichment_table[:topk, 2], dtype=float)
    
    ax.bar(x, heights, linewidth=0)
    ax.set_xticklabels(x, rotation=90, fontsize=6)
    ax.set_title(title)
    ax.set_ylabel('Enrichment score')
    # ax.set_xlabel('Pathway/GO term')
    
    fig.tight_layout()
    fig.savefig('{}/GO_enrich.png'.format(base))

def empirical_FDR(NES, null_NES, pw_name, base, fdr=0.05):
    
    Ts = np.sort(null_NES)[-1000:]

    # f = Fitter(NES)
    # f.fit()
    # print('NES dist: {}'.format(f.summary()))

    # f = Fitter(null_NES)
    # f.fit()
    # print('null_NES dist: {}'.format(f.summary()))
    # if pw_name == 'KEGG' or pw_name == 'REACTOME':
    #     dist = hypsecant
    # else:
    #     dist = norm
    dist = hypsecant

    # s_o, loc_o, scale_o = dist.fit(NES)
    a, b = dist.fit(null_NES)
    
    p_vals = np.array(list(map(lambda x: dist.sf(x, a, b), NES)))
    hit_idxs = p_vals < fdr
    least_sig_hit_NES = NES[np.argmax(p_vals[hit_idxs])]

    plt.hist(NES, bins=50, color='blue', alpha=0.3, 
        density=True, label='Obs')
    # plt.plot(np.sort(NES), dist.pdf(np.sort(NES), s_o, loc_o, scale_o), color='blue')
    plt.hist(null_NES, bins=50, color='red', alpha=0.3, 
        density=True, label='Null')
    plt.plot(np.sort(null_NES), dist.pdf(np.sort(null_NES), a, b), color='red')
    plt.axvline(least_sig_hit_NES, linestyle=':', linewidth='1', color='k')
    plt.xlabel('Enrichment score')
    plt.ylabel('Density')
    # plt.title('{} FDR = {:.2f}, s_obs = {}'.format(
    #     pw_name, emp_fdr[best_arg], np.count_nonzero(hit_idxs)))
    plt.title('{} FDR = {:.2f}, s_obs = {}'.format(
        pw_name, fdr, np.count_nonzero(hit_idxs)))
    plt.legend(loc='upper right')
    plt.savefig('{}/{}_hist.svg'.format(base, pw_name))
    plt.close()

    return p_vals

def pw_overlap(set_dict, pw, base):
    data = from_contents(set_dict)
    fig = plt.Figure()
    UpSet(data, sort_by='cardinality', show_counts=True).plot(fig=fig)
    fig.suptitle(pw)
    fig.tight_layout()
    fig.savefig('{}/upset_plots/MatPat_UpSet_{}.png'.format(base, pw))
    plt.close()

def run_GSEA(exp_name:str, fdr=0.05):
    global PHENO 
    alpha = 1
    pheno = PHENO
    
    for pw in ['KEGG', 'REACTOME', 'WP']:
        rank_labels = ['logP']
        # rank_labels = ['Acc']
        for rank in rank_labels:
            pw_set_names = []
            pw_sets = []
            
            path = f'../results_{exp_name}/enrichments/pathways/{pw}_enrichment.txt'
            if not os.path.exists(path):
                res_path = f'../results_{exp_name}/{PHENO}_Loss_{exp_name}_gene_summary.csv'
                GSEA_pathway_enrichment(
                    res_path=res_path, 
                    enr_res_path=path,
                    pheno=pheno,
                    rank=rank,
                    pw_subset=[pw,], 
                    alpha=alpha,
                    col='entrez_id')
            
            df = pd.read_csv(path, sep='\t')
            NES = df['NES'].values
            null_NES = np.load(path.replace('txt', 'npy'))
            
            base = '/'.join(path.split('/')[:-1])
            p_vals = empirical_FDR(NES, null_NES, pw, base, fdr=fdr)
            df['p'] = p_vals

            if df.columns[0] == 'FDR':
                df['FDR'] = 0
                df.loc[df['p'] < fdr, 'FDR'] = 1
            else:
                df['FDR'] = 0
                df.loc[df['p'] < fdr, 'FDR'] = 1
                cols = df.columns[:-1]
                cols = cols.insert(0, 'FDR')
                df = df[cols]

            df.to_csv(path, sep='\t', index=False)
                
def run_T2D(fdr=0.05):
    global PHENO
    dfp = '/home/upamanyu/GWASOnSteroids/Enrichment_scripts'\
            '/manual_enrich_{}_rank_{}/{}_GSEA_{}_enrich.{}'
    alpha = 1
    pheno = PHENO
    for pw in ['KEGG', 'REACTOME', 'WP']:
        rank_labels = ['logP']
        for rank in rank_labels:
            pw_set_names = []
            pw_sets = []
            
            path = dfp.format(pheno, rank, pheno, pw, 'txt')
            enrich_fold = '/'.join(path.split('/')[:-1])
            
            if not os.path.exists(enrich_fold):
                os.mkdir(enrich_fold)

            if not os.path.exists(path):
                
                res_path ='/home/upamanyu/GWASOnSteroids/Results/'\
                    'Annotated_NoDup_{}_NN_GWAS.csv'.format(pheno)
                # res_path = '/home/upamanyu/GWASOnSteroids/Results/Annotated_AD_Meta.csv'
                # res_path ='/home/upamanyu/GWASOnSteroids/Results/Annotated_NoDup_PatAD_NN_GWAS.csv'
                
                GSEA_pathway_enrichment(
                    res_path=res_path, 
                    pheno=pheno,
                    rank=rank,
                    pw_subset=[pw,], 
                    alpha=alpha,
                    col='entrez_id')
                
            df = pd.read_csv(path, sep='\t')
            NES = df['NES'].values
            null_NES = np.load(path.replace('txt', 'npy'))
            
            base = '/'.join(path.split('/')[:-1])
            p_vals = empirical_FDR(NES, null_NES, pw, base, fdr=fdr)
            df['p'] = p_vals

            if df.columns[0] == 'FDR':
                df['FDR'] = 0
                df.loc[df['p'] < fdr, 'FDR'] = 1
            else:
                df['FDR'] = 0
                df.loc[df['p'] < fdr, 'FDR'] = 1
                cols = df.columns[:-1]
                cols = cols.insert(0, 'FDR')
                df = df[cols]
                
            df.to_csv(path, sep='\t', index=False)

def run_STR(res_path, pheno, fdr=0.05):
    dfp = '/home/upamanyu/Will_analysis/manual_enrich_{}_rank_{}/'\
            '{}_GSEA_{}_enrich.{}'
    enr_fold = '/'.join(dfp.split('/')[:-1]).format(pheno, 'P')
    if not os.path.exists(enr_fold):
        os.mkdir(enr_fold)
    
    go_subset = None
    alpha = 1
    GSEA_pathway_enrichment(
        res_path=res_path, 
        pw_subset=['KEGG', 'REACTOME', 'WP'], 
        pheno=pheno, 
        alpha=alpha, 
        col='new_entrez_id')
    GSEA_GO_enrichment(
        res_path=res_path, 
        pheno=pheno, 
        go_subset=go_subset, 
        alpha=alpha, 
        col='new_gene')
    
    for pw in ['KEGG', 'REACTOME', 'WP', 'GO']:
        rank_labels = ['P']
        for rank in rank_labels:
            path = dfp.format(pheno, rank, pheno, pw, 'txt')
            df = pd.read_csv(path, sep='\t')
            
            NES = df['NES'].values
            null_NES = np.load(path.replace('txt', 'npy'))
            null_NES = np.quantile(null_NES, np.arange(1, len(NES)+1)/len(NES))
            base = '/'.join(path.split('/')[:-1])
            sig_args = empirical_FDR(NES, null_NES, pw, base, fdr=fdr)
            
            if df.columns[0] == 'FDR':
                df['FDR'] = 0
                df.loc[sig_args, 'FDR'] = 1
            else:
                df['FDR'] = 0
                df.loc[sig_args, 'FDR'] = 1
                cols = df.columns[:-1]
                cols = cols.insert(0, 'FDR')
                df = df[cols]
                
            df.to_csv(path, sep='\t', index=False)

def GO_group_enr(regions, fdr=0.05):
    go_subsets = ['biological_process', 'cellular_component', 
                    'molecular_function']
    
    for reg in regions:
        enr_fold = '/home/upamanyu/Will_analysis/'\
                'manual_enrich_{reg}_rank_P'.format(reg=reg)
        enr_path = '{enr_fold}/{reg}_GSEA_GO_enrich.txt'.format(
            enr_fold=enr_fold, reg=reg)
        enr_df = pd.read_csv(enr_path, sep='\t')
        null_dist = np.load(enr_path.replace('.txt', '.npy'))
        null_dist = null_dist.reshape((-1, 1000))
        go_ids= enr_df['GO_id'].to_list()
        
        for go_sub_name in go_subsets:
            go_sub_path = '/home/upamanyu/Will_analysis/'\
                'data/GO_data/{go_sub}_subterms.txt'.format(
                    go_sub=go_sub_name)
            with open(go_sub_path, 'r') as f:
                glist = f.read().strip('\n').split('\n')

            idxs = np.where(np.isin(glist, go_ids))[0]
            sub_enr_df = enr_df.iloc[idxs, :]
            sub_null_dist = null_dist[idxs].flatten()

            NES = sub_enr_df['NES'].values
            null_NES = np.quantile(sub_null_dist, np.arange(1, len(NES)+1)/len(NES))
            sig_args = empirical_FDR(
                NES=NES, 
                null_NES=null_NES, 
                pw_name='GO_{}'.format(go_sub_name), 
                base=enr_fold,
                fdr=fdr)
            sub_enr_df['FDR'] = 0
            sub_enr_df.loc[sig_args, 'FDR'] = 1
            
            sub_enr_path = '{enr_fold}/'\
                '{reg}_GSEA_GO_{go_sub}_enrich.txt'.format(
                    enr_fold=enr_fold, reg=reg, go_sub=go_sub_name)
            print(len(idxs), sub_enr_path)
            sub_enr_df.to_csv(sub_enr_path, sep='\t', index=False)
            np.save(sub_enr_path.replace('.txt', '.npy'), sub_null_dist)
        
def enrichment_heatmap(regions, rank):
    
    pw_sets = ['KEGG', 'REACTOME', 'WP']
    # pw_sets.extend(['GO_biological_process', 'GO_molecular_function', 'GO_cellular_component'])
    # pw_sets = ['GO']
    
    for i, pw in enumerate(pw_sets):
        # fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        pw_NES_df = pd.DataFrame()
        for reg in regions:
            enr_fold = '../manual_enrich_{reg}_rank_{rank}'.format(
                reg=reg, rank=rank)
            enr_path = '{enr_fold}/{reg}_GSEA_{pw}_enrich.txt'.format(
                enr_fold=enr_fold, reg=reg, pw=pw)
            enr_df = pd.read_csv(enr_path, sep='\t')
            # enr_df['NES'] = np.where(enr_df['FDR'].values == 1, 
            #                         enr_df['NES'].values, 
            #                         np.zeros(enr_df['NES'].shape))
            enr_df.rename(columns={'NES':reg}, inplace=True)
            
            if 'GO' in pw:
                enr_df.set_index(enr_df.columns[4], inplace=True)
            else:
                enr_df.set_index(enr_df.columns[3], inplace=True)
            if len(pw_NES_df) == 0:
                pw_NES_df = enr_df.iloc[:, 2:3]
                pw_NES_df['FDR'] = enr_df['FDR'].values
            else:
                pw_NES_df = pw_NES_df.merge(enr_df.iloc[:, 2:3], 
                    left_index=True, right_index=True)
                pw_NES_df['FDR'] = pw_NES_df['FDR'].values & \
                    enr_df.loc[pw_NES_df.index]['FDR'].values
        
        pw_FDR = pw_NES_df['FDR'].values
        pw_NES_df.drop(columns=['FDR'], inplace=True)
        # pw_names = np.array(['_'.join(p.split('_')[1:]) for p in pw_NES_df.index.values])
        pw_names = pw_NES_df.index.values
        pw_NES_df.index = np.arange(len(pw_NES_df))

        cmap = sns.clustermap(pw_NES_df, standard_scale=1, 
                row_cluster=True, col_cluster=False, 
                cmap='Reds')
        cmap.cax.set_visible(False)
        cmap.ax_col_dendrogram.set_visible(False)

        pw_NES_df = pw_NES_df.loc[cmap.dendrogram_row.reordered_ind]
        pw_FDR = pw_FDR[cmap.dendrogram_row.reordered_ind]
        pw_names = pw_names[cmap.dendrogram_row.reordered_ind]
        sig_idxs = np.where(pw_FDR == 1)[0]
        with open('../heatmaps_{}/{}_all_FDR.txt'.format(rank, pw), 'w') as f:
            f.write('\n'.join(list(pw_names[sig_idxs])))
        
        cmap.ax_heatmap.set(yticks=[], yticklabels=[], title=pw)
        cmap.savefig('../heatmaps_{}/{}_Enrich_heatmap.png'.format(rank, pw))
        plt.close()
            
def hit_overlap_pws(fdr):
    global PHENO

    pheno = PHENO
    
    ndf = pd.read_csv('/home/upamanyu/GWASOnSteroids/Results/{}_Top_SNPs.csv'.format(pheno))
    # ndf = pd.read_csv('/home/upamanyu/GWASOnSteroids/Results/{}_suggestive.csv'.format(pheno))
    ndf = ndf[['Gene']]
    gpath = '/home/upamanyu/GWASOnSteroids/GWASNN/datatables/gene_list2.csv'
    gdf = pd.read_csv(gpath)
    gdf = gdf.loc[gdf['chrom'].isin([str(c) for c in range(1, 23)])]

    ndf = ndf.merge(gdf, left_on='Gene', right_on='symbol')
    print(ndf.columns)

    ngs = ndf['entrez_id'].to_list()
    ngs = [str(int(g)) for g in ngs]
    pw_dict = get_pathway_dict()

    s = 'pw_id,pw_name,NES,FDR,hit_coverage,hit_genes\n'
    for pw_set in ['KEGG', 'REACTOME', 'WP']:
        pwdf = pd.read_csv('../manual_enrich_{pheno}_rank_logP/'\
            '{pheno}_GSEA_{pw_set}_enrich.txt'.format(
                pheno=pheno, pw_set=pw_set), sep='\t')
        pwdf = pwdf.loc[pwdf['p'] < fdr]

        pws = pwdf['pathway'].to_list()
        hit_pw_dict = {}
        for pw in tqdm(pws):
            gset = [str(int(g)) for g in pw_dict[pw]]
            overlap_gs = [float(eid) for eid in list(set(ngs).intersection(set(gset)))]
            overlap_gs = gdf.set_index('entrez_id').loc[overlap_gs]['symbol'].to_list()
            hit_pw_dict[pw] = overlap_gs
            enr_p = pwdf.loc[pwdf['pathway'] == pw]['p'].values[0]
            NES = pwdf.loc[pwdf['pathway'] == pw]['NES'].values[0]
            s += '{},{},{},{},{}/{},{}\n'.format(
                    get_pathway_ID(pw), pw, NES, enr_p, 
                    len(overlap_gs), len(gset), '+'.join(overlap_gs))
        

    with open('../manual_enrich_{}_rank_logP/enriched_pws.csv'.format(pheno), 'w') as f:
        f.write(s)
            
def filter_Reactome():
    global PHENO
    pheno = PHENO

    edges = pd.read_csv('/mnt/sdb/Pathway_data/Reactome/ReactomePathwaysRelation.txt', sep='\t', header=None)
    edges = edges.loc[edges[0].str.startswith('R-HSA')]
    edges = edges.loc[edges[1].str.startswith('R-HSA')]
    edges = [(edges.iloc[i, 0], edges.iloc[i, 1]) for i in range(len(edges))]
    
    hits = pd.read_csv('/home/upamanyu/GWASOnSteroids/Enrichment_scripts/manual_enrich_{}_rank_logP/enriched_pws.csv'.format(pheno))
    reac_hits = hits.loc[hits['pw_id'].str.startswith('R-HSA')]
    hit_nodes = reac_hits['pw_id'].to_list()

    G = nx.DiGraph()
    G.add_edges_from(edges)
    desc_paths = []
    for hn in hit_nodes:
        desc = nx.algorithms.dag.descendants(G, hn)
        desc_paths.extend(desc)
        print('{:<15}{}'.format(hn, [d for d in desc if d in hit_nodes]))
    
    final_hits = set(hit_nodes).difference(set(desc_paths))

    hits = hits.loc[(hits['pw_id'].isin(final_hits)) | ~(hits['pw_id'].str.startswith('R-HSA'))]
    hits.to_csv('../manual_enrich_{}_rank_logP/enriched_pws_reactome_0indegree.csv'.format(pheno), index=False)

def get_pathway_ID(pw_name):
    url = "https://www.gsea-msigdb.org/gsea/msigdb/cards/{}.html"
    url = url.format(pw_name)
    
    html_response = requests.get(url=url)
    soup = BeautifulSoup(html_response.text, 'html.parser')
    
    return soup.find('th', string='Exact source').parent.td.text

def start_enrichment():
    global PHENO
    PHENO = 'FH_AD'
    run_GSEA(exp_name='Sens8_00_GS10_v4', fdr=0.01)
    # hit_overlap_pws(fdr=0.01)
    # filter_Reactome()

if __name__ == '__main__':
    # filter_pathway_dict()
    start_enrichment()
