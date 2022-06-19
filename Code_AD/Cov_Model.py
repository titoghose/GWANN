import sys
sys.path.append('.')

import argparse
from GWANN.train_model_covs import *
from GWANN.models import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str)
    args = parser.parse_args()
    label = args.label

    for att in [AttentionMask1]:

        s = datetime.datetime.now()
        with open('./params/gene_subsets.yaml', 'r') as f:
            g_yaml = yaml.load(f, Loader=yaml.FullLoader)

        genes_df = pd.read_csv('../GWASNN/datatables/genes.csv')
        genes_df.drop_duplicates(['symbol'], inplace=True)
        genes_df.set_index('symbol', drop=False, inplace=True)

        gene_list = ['APOE', 'APOC1', 'TOMM40', 'APP', 'CLU', 'BCAM', 'PICALM']
        print(len(gene_list))

        df = genes_df.loc[gene_list]
        df.drop_duplicates(['symbol'], inplace=True)
        df = df.astype({'chrom':int})
        df.sort_values(['num_snps'], inplace=True)

        exp_name = 'AD_{}_Cov'.format(label.split('_')[0])
        exp = Experiment(exp_name, label, 
            '/home/upamanyu/GWASOnSteroids/Code_AD/params', 2500)
        exp.gene_type = ''
        exp.GPU_LIST = list(np.repeat([0, 1, 2, 3, 4, 5, 6], 1))

        # Setting the model for the Experiment
        model = GroupAttention
        model_params = {
            'grp_size':10,
            'inp':0,
            'enc':0,
            # 'num_snps':0,
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
            'epochs': 2000,
        }
        exp.set_model(model, model_params)
        exp.hyperparam_dict = hp_dict
        exp.set_paths()
        exp.perm_batch_size = -1

        genes = {'names':[], 'ids':[], 'chrom':[]}
        genes['names'] = [[n,] for n in df['symbol'].values[0:]]
        genes['chrom'] = [[c,] for c in df['chrom'].values[0:]]
        genes['ids'] = [[i,] for i in df['id'].values[0:]]
        print(genes)

        exp.permloop(genes, 1e3)
            
        e = datetime.datetime.now()
        print('\n\n', (e-s))

