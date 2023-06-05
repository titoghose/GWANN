# coding: utf-8
import sys

sys.path.append('/home/upamanyu/GWANN')

import datetime
import argparse

import numpy as np
import pandas as pd
import torch.nn as nn

from GWANN.models import AttentionMask1, GroupAttention
from GWANN.train_model_covs import Experiment

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str, required=True)
    args = parser.parse_args()
    label = args.label

    for att in [AttentionMask1]:

        s = datetime.datetime.now()
        genes_df = pd.read_csv('/home/upamanyu/GWANN/GWANN/datatables/gene_annot.csv')
        genes_df.drop_duplicates(['symbol'], inplace=True)
        genes_df.set_index('symbol', drop=False, inplace=True)

        gene_list = ['BCR', 'RBFOX2']
        
        df = genes_df.loc[gene_list]
        df = df.astype({'chrom':str})

        exp_name = f'{label}_Cov'
        # Setting the model for the Experiment
        model = GroupAttention
        model_dict = {
            'grp_size':10,
            'inp':0,
            'enc':0,
            'h':[128, 64, 16],
            'd':[0.3, 0.3, 0.3],
            'out':2,
            'activation':nn.ReLU,
            'att_model':att, 
            'att_activ':nn.Sigmoid
        }
        hp_dict = {
            'optimiser': 'adam',
            'lr': 1e-4,
            'batch': 256,
            'epochs': 50,
        }
        exp = Experiment(prefix=exp_name, 
                label=label, 
                params_base='/home/upamanyu/GWANN/Code_AD/params/reviewer_rerun', 
                buffer=2500, model=model, model_dict=model_dict, hp_dict=hp_dict, 
                gpu_list=list(np.repeat([0, 1], 1)))

        genes = {'names':[], 'chrom':[], 'start':[], 'end':[]}
        genes['names'] = df['symbol'].to_list()
        genes['chrom'] = df['chrom'].to_list()
        genes['start'] = df['start'].to_list()
        genes['end'] = df['end'].to_list()
        print(genes)

        exp.parallel_run(genes)
            
        e = datetime.datetime.now()
        print('\n\n', (e-s))

