import sys

sys.path.append('/home/upamanyu/GWANN')

import numpy as np
import pandas as pd

from run_genes import model_pipeline, create_csv_data

def sensitivity_1(chroms:list):
    param_folder = '/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun'
    
    mat_df = pd.read_csv(f'{param_folder}/Annotated_NoDup_MatAD_NN_GWAS.csv')
    pat_df = pd.read_csv(f'{param_folder}/Annotated_NoDup_PatAD_NN_GWAS.csv')
    meta_df = pd.read_csv(f'{param_folder}/Annotated_AD_Meta.csv')
    labels = {'PATERNAL_MARIONI':meta_df, 
              'MATERNAL_MARIONI':meta_df}
    for label, df in labels.items():
        hits = df.loc[df['P'] < (0.05/73310)].astype({'chrom':str})
        print(label)
        for chrom, idx in hits.groupby('chrom').groups.items():
            if not str(chrom) in chroms:
                continue
            glist = hits.loc[idx]['Gene'].to_list()
            print(chrom, glist)
            create_csv_data(label=label, param_folder=param_folder, 
                            chrom=chrom, glist=glist, split=True)

    # gpu_list = list(np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
    # model_pipeline(exp_name='Sens1', label='MATERNAL_MARIONI', 
    #                param_folder=param_folder, gpu_list=gpu_list)
    # model_pipeline(exp_name='Sens1', label='PATERNAL_MARIONI', 
    #                param_folder=param_folder, gpu_list=gpu_list)

if __name__ == '__main__':
    chroms = sys.argv[1].split(',')
    sensitivity_1(chroms)
