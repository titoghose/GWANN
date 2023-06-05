import sys
sys.path.append('.')

import argparse
from GWANN.train_model_LG import *
from GWANN.models import GWANNet5, AttentionMask1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type=int, default=0)
    parser.add_argument('--label', type=str)
    parser.add_argument('--chrom', type=str)
    args = parser.parse_args()
    label = args.label

    for att in [AttentionMask1]:

        s = datetime.datetime.now()
        with open('./params/gene_subsets.yaml', 'r') as f:
            g_yaml = yaml.load(f, Loader=yaml.FullLoader)

        genes_df = pd.read_csv('/home/upamanyu/GWASOnSteroids/GWASNN/datatables/genes.csv')
        genes_df.drop_duplicates(['symbol'], inplace=True)
        genes_df.set_index('symbol', drop=False, inplace=True)

        df = genes_df.loc[genes_df['chrom'] == args.chrom]
        df = df.astype({'chrom':int})
        df.sort_values(['num_snps'], ascending=False, inplace=True)

        exp_name = label + '_Chr' + args.chrom
        exp = Experiment(exp_name, label, 
            '/home/upamanyu/GWASOnSteroids/Code_AD/params', 50, 20000)
        exp.gene_type = ''
        exp.GPU_LIST = list(np.repeat([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2))

        # Setting the model for the Experiment
        model = GWANNet5
        model_params = {
            'grp_size':10,
            'enc':0,
            'num_snps':0,
            'h':[128, 64],
            'd':[0.3, 0.3],
            'out':16,
            'activation':nn.ReLU,
            'att_model':att, 
            'att_activ':nn.Sigmoid
        }
        hp_dict = {
            'optimiser': 'adam',
            'lr': 1e-4,
            'batch': 256,
            'epochs': 500,
        }
        exp.set_model(model, model_params)
        exp.hyperparam_dict = hp_dict
        exp.set_paths()
        exp.perm_batch_size = -1
        exp.SNP_THRESH = 10000
        
        genes = {'names':[], 'ids':[], 'chrom':[]}
        genes['names'] = [[n,] for n in df['symbol'].values[0:]]
        genes['chrom'] = [[c,] for c in df['chrom'].values[0:]]
        genes['ids'] = [[i,] for i in df['id'].values[0:]]
        print(len(genes['names']))

        # Split genes into windows 
        snp_cnt = pd.read_csv('../Runs/num_snps.csv')
        snp_cnt.drop_duplicates(['Gene', 'bp'], inplace=True)
        snp_cnt.set_index('Gene', inplace=True, drop=False)

        gws, cws, iws = [], [], []
        for gi, g in enumerate(genes['names']):
            snp_win = exp.snp_win
            try:
                num_snps = snp_cnt.loc[g[0]]['num_snps']
            except:
                continue
            if num_snps > exp.SNP_THRESH:
                continue
            num_win = int(np.ceil(num_snps/snp_win))
            remaining_snps = num_snps

            for win in range(num_win):
                sind = win * snp_win
                eind = sind+remaining_snps if remaining_snps < snp_win else (win+1)*snp_win
                nsnps = eind-sind
                gn = '{}_{}'.format(g[0], str(win))
                gws.append([gn,])
                cws.append(genes['chrom'][gi])
                iws.append(genes['ids'][gi])

        genes['names'] = gws[0:]
        genes['chrom'] = cws[0:]
        genes['ids'] = iws[0:]
        print(len(genes['names']))

        exp.permloop(genes, 1e3)

        e = datetime.datetime.now()
        print('\n\n', (e-s))
