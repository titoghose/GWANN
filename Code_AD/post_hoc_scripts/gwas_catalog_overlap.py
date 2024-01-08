import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def split_into_list(r) -> list:
    glist = [r['MAPPED_GENE']] + [r['REPORTED GENE(S)']]
    glist = ' - '.join(glist).split(' - ')
    glist = ' x '.join(glist).split(' x ')
    glist = '; '.join(glist).split('; ')
    glist = ', '.join(glist).split(', ')
    return glist

def group_AD_related_traits(gwas_all_assoc) -> pd.DataFrame:
    gwas_AD_assoc = gwas_all_assoc.loc[
            gwas_all_assoc['DISEASE/TRAIT'].str.contains('Alzheimer', case=False) |
            gwas_all_assoc['DISEASE/TRAIT'].str.contains('Neurofibrillary tangles', case=False) |
            gwas_all_assoc['DISEASE/TRAIT'].str.contains('p-tau', case=False) |
            gwas_all_assoc['DISEASE/TRAIT'].str.contains('t-tau', case=False) |
            gwas_all_assoc['DISEASE/TRAIT'].str.contains('PHF-tau', case=False)
            ].copy()

    gwas_AD_assoc.loc[gwas_AD_assoc['DISEASE/TRAIT'].str.contains('Alzheimer', case=False), 'DISEASE/TRAIT'] = 'Alzheimer'
    gwas_AD_assoc.loc[gwas_AD_assoc['DISEASE/TRAIT'].str.contains('Neurofibrillary tangles', case=False), 'DISEASE/TRAIT'] = 'Neurofibrillary tangles'
    gwas_AD_assoc.loc[
        gwas_AD_assoc['DISEASE/TRAIT'].str.contains('p-tau', case=False) |
        gwas_AD_assoc['DISEASE/TRAIT'].str.contains('t-tau', case=False) |
        gwas_AD_assoc['DISEASE/TRAIT'].str.contains('PHF-tau', case=False), 'DISEASE/TRAIT'] = 't-tau, p-tau, PHF-tau'
    
    return gwas_AD_assoc


overlap_path = '../results_Sens8_v4/enrichments/gwas_catalog_overlap.csv'
if not os.path.exists(overlap_path):
    gwas_all_assoc = pd.read_csv('/mnt/sdb/GWAS_Catalog_AD/gwas_catalog_v1.0-associations_e110_r2023-07-29.tsv', 
                                sep='\t', dtype={'MAPPED_GENE':str, 'REPORTED GENE(S)':str}, 
                                keep_default_na=False)

    # Alzheimer's related trait grouping 
    gwas_AD_assoc = group_AD_related_traits(gwas_all_assoc)
    gwas_AD_assoc['P-VALUE'] = gwas_AD_assoc['P-VALUE'].apply(lambda x:float(x)).values
    gwas_AD_assoc = gwas_AD_assoc.loc[gwas_AD_assoc['P-VALUE'] < 1e-5]
    gwas_AD_assoc['GENE_LIST'] = gwas_AD_assoc.apply(lambda x: split_into_list(x), axis=1)
    
    gwas_AD_assoc = gwas_AD_assoc.set_index(['DISEASE/TRAIT', 'P-VALUE', 'PUBMEDID'])['GENE_LIST'].apply(pd.Series).stack().reset_index().rename(columns={0:'Gene'})
    gwas_AD_assoc['P-VALUE'] = gwas_AD_assoc['P-VALUE'].astype(float)
    gwas_AD_assoc = gwas_AD_assoc[['DISEASE/TRAIT', 'P-VALUE', 'Gene', 'PUBMEDID']]
    
    group_df = gwas_AD_assoc.sort_values(['DISEASE/TRAIT', 'Gene', 'P-VALUE'])
    group_df.drop_duplicates(['DISEASE/TRAIT', 'Gene'], inplace=True)
    missing_gene_values = ['No mapped genes', 'Intergenic', 'intergenic', 'N/A', 'NR', 'None', 'NA']
    group_df = group_df.loc[~group_df['Gene'].isin(missing_gene_values,)]
    group_df = group_df.loc[~group_df['Gene'].isin(missing_gene_values)]
    group_df = group_df.loc[group_df['P-VALUE'] < 5e-8]
    group_df.to_csv('gwas_catalog_AD_groups.csv', index=False)
    print('hello')

    # Overlap with GWANN hits
    nn_AD_hits = pd.read_csv('/home/upamanyu/GWANN/Code_AD/results_Sens8_v4/LD/pruned_gene_hits_1e-23.csv')
    nn_AD_hits = nn_AD_hits.loc[~nn_AD_hits['pruned']]
    nn_AD_genes = nn_AD_hits['Gene'].to_list()

    gwas_AD_overlap = gwas_AD_assoc.loc[gwas_AD_assoc['Gene'].isin(nn_AD_genes)]
    gwas_AD_overlap = gwas_AD_overlap.loc[gwas_AD_overlap['P-VALUE'] < 1e-5]
    gwas_AD_overlap.sort_values(['Gene', 'DISEASE/TRAIT', 'P-VALUE'], 
                                ascending=True, inplace=True)
    gwas_AD_overlap.drop_duplicates(['Gene', 'DISEASE/TRAIT', 'PUBMEDID'], inplace=True)
    gwas_AD_overlap = gwas_AD_overlap.groupby(['Gene', 'DISEASE/TRAIT'])['PUBMEDID'].count().reset_index() 
    
    gwas_AD_overlap = gwas_AD_overlap.pivot_table(index='Gene', columns='DISEASE/TRAIT', 
                                                values='PUBMEDID')
    gwas_AD_overlap.fillna(0, inplace=True)
    print(gwas_AD_overlap.head(10))

    # Add non-overlapping genes
    non_overlapping = set(nn_AD_genes).difference(set(gwas_AD_overlap.index.to_list()))
    for g in non_overlapping:
        gwas_AD_overlap.loc[g] = [0, 0, 0]
    gwas_AD_overlap.reset_index().to_csv(overlap_path, index=False)

else:
    gwas_AD_overlap = pd.read_csv(overlap_path, index_col=0)

# Heatmap 
gwas_AD_overlap[gwas_AD_overlap == 0] = np.nan
gwas_AD_overlap.sort_values(['Alzheimer', 'Neurofibrillary tangles', 't-tau, p-tau, PHF-tau'], 
                            ascending=False, inplace=True)
gwas_AD_overlap = gwas_AD_overlap.T
fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax.tick_params(labelsize=7)
sns.heatmap(data=gwas_AD_overlap, ax=ax, xticklabels=True, cmap='Reds', annot=True)
ax.collections[0].colorbar.set_label("Number of GWAS", fontsize=8)
plt.show()


# colorbar = ax.collections[0].colorbar
# colorbar.set_ticks([0, 1, 2]) 
# colorbar.set_ticklabels(['No evidence', 'P<1e-5', 'P<5e-8'])

fig.tight_layout()
fig.savefig('../results_Sens8_v4/enrichments/gwas_catalog_overlap.svg')
fig.savefig('../results_Sens8_v4/enrichments/gwas_catalog_overlap.png', dpi=100)
plt.close()
