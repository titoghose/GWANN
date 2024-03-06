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

    gwas_AD_assoc.loc[gwas_AD_assoc['DISEASE/TRAIT'].str.contains('Alzheimer', case=False), 'DISEASE/TRAIT'] = 'Alzheimer\'s disease'
    gwas_AD_assoc.loc[
        gwas_AD_assoc['DISEASE/TRAIT'].str.contains('Neurofibrillary tangles', case=False) |
        gwas_AD_assoc['DISEASE/TRAIT'].str.contains('p-tau', case=False) |
        gwas_AD_assoc['DISEASE/TRAIT'].str.contains('t-tau', case=False) |
        gwas_AD_assoc['DISEASE/TRAIT'].str.contains('PHF-tau', case=False), 'DISEASE/TRAIT'] = 'Neurofibrillary tangles or \ntau protein meausurement'
    
    return gwas_AD_assoc

os.chdir('/home/upamanyu/GWANN/Code_AD/results_Sens8_v4/')
overlap_path = 'enrichments/gwas_catalog_overlap_1e-25.csv'
if not os.path.exists(overlap_path):
    gwas_all_assoc = pd.read_csv('/mnt/sdb/GWAS_Catalog_AD/gwas_catalog_v1.0-associations_e111_r2024-01-19.tsv', 
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
    nn_AD_hits = pd.read_csv('LD/pruned_gene_hits_1e-25.csv')
    nn_AD_hits = nn_AD_hits.loc[~nn_AD_hits['pruned']]
    nn_AD_genes = nn_AD_hits['Gene'].to_list()

    gwas_AD_overlap = gwas_AD_assoc.loc[gwas_AD_assoc['Gene'].isin(nn_AD_genes)]
    gwas_AD_overlap = gwas_AD_overlap.loc[gwas_AD_overlap['P-VALUE'] < 5e-8]
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
        gwas_AD_overlap.loc[g] = [0]*len(gwas_AD_overlap.columns)
    gwas_AD_overlap.reset_index().to_csv(overlap_path, index=False)

else:
    gwas_AD_overlap = pd.read_csv(overlap_path, index_col=0)

# Heatmap 
gwas_AD_overlap[gwas_AD_overlap == 0] = np.nan
gwas_AD_overlap.sort_values(['Alzheimer\'s disease', 'Neurofibrillary tangles or \ntau protein meausurement'], 
                            ascending=False, inplace=True)
gwas_AD_overlap = gwas_AD_overlap
fig, ax = plt.subplots(1, 1, figsize=(4.6, 10))
sns.heatmap(data=gwas_AD_overlap, ax=ax, yticklabels=True, cmap='Reds', 
            annot=True)
for t in ax.texts:
    t.set_fontsize(16)
ax.set_ylabel('Genes', fontsize=16)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
cbar = ax.collections[0].colorbar
cbar.set_label("Number of GWAS", fontsize=16)
cbar.ax.tick_params(labelsize=16)

with open('enrichments/gwas_catalog_overlap_ytick_order.txt', 'w') as f:
    f.write('\n'.join([l.get_text() for l in ax.get_yticklabels()]))

plt.show()

fig.tight_layout()
fig.savefig('enrichments/gwas_catalog_overlap_1e-25.svg')
fig.savefig('enrichments/gwas_catalog_overlap_1e-25.png', dpi=300)
plt.close()
