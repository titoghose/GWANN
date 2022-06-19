# coding: utf-8
import os
import sys
import csv
import yaml
import tqdm
import pickle
import datetime
import traceback
import numpy as np
import pandas as pd 
from functools import partial
import warnings
warnings.filterwarnings('ignore')

from GWANN.train_utils import *
from GWANN.dataset_utils import *
from GWANN.models import *

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch
import torch.nn as nn

class Experiment:
    def __init__(self, pre, label, params_base, bp=0):

        # Experiment descriptive parameters
        self.PRE = pre
        self.label = label
        self.bp = bp        
        self.params_base = params_base

        # Base parameter YAML files
        self.sys_params = None
        self.covs = None
        self.set_params()

        # Path parameters
        self.RUN_FOLDER = ''
        self.PERM_FILE = ''
        self.PERM_ERR_FILE = ''
        self.TRAIN_FILE = ''
        self.TRAIN_ERR_FILE = ''
        self.NONE_DATA_FILE = ''
        self.MODEL_SNP_LIMIT = 50

        # NN model and permutation test specific parameters
        self.model = None
        self.model_params = None
        self.model_dir = ''
        self.hyperparam_dict = None
        self.ptest_f = ''
        self.perms = 0
        self.perm_batch_size = 1024

        # Dataset generation parameters
        self.OVER_COEFF = 0.0
        self.BALANCE = 1.0
        self.SNP_THRESH = 1000
        self.GENOTYPES = {}
        self.PHEN_COV = None
        self.train_oversample = 10
        self.test_oversample = 10
        
        # Training parameters
        self.GPU_LIST = []
        self.cov_encodings = None#self.load_cov_encodings()
        self.gene_type = ''
    
    def set_params(self):
        """Load the YAML file containing system specific paths and 
        parameters.
        """
        sysf = '{}/params_{}.yaml'.format(self.params_base, self.label)
        with open(sysf, 'r') as params_file:
            self.sys_params = yaml.load(params_file, Loader=yaml.FullLoader) 
        covsf = '{}/covs_{}.yaml'.format(self.params_base, self.label)
        with open(covsf, 'r') as covs_file:
            self.covs = yaml.load(covs_file, Loader=yaml.FullLoader)
        self.covs = self.covs['COVARIATES']
        
    def set_paths(self):
        """Function to set all file and folder paths, later used by
        other functions.
        """
        self.RUN_FOLDER = '{}/{}'.format(self.sys_params['RUNS_BASE_FOLDER'], self.PRE)
        self.PERM_FILE = '{}/{}'.format(self.RUN_FOLDER, 'perm.txt')
        self.PERM_ERR_FILE = '{}/{}'.format(self.RUN_FOLDER, 'perm_err.txt')
        self.TRAIN_FILE = '{}/{}'.format(self.RUN_FOLDER, 'train.txt')
        self.TRAIN_ERR_FILE = '{}/{}'.format(self.RUN_FOLDER, 'train_err.txt')
        self.NONE_DATA_FILE = '{}/{}'.format(self.RUN_FOLDER, 'None_data.txt')
        
        hp_dict = self.hyperparam_dict
        if not os.path.isdir(self.RUN_FOLDER):
            os.mkdir(self.RUN_FOLDER)
        
        model_train_params = 'LR:{}_BS:{}_Optim:{}'.format(
            hp_dict['lr'], 
            hp_dict['batch'], 
            hp_dict['optimiser'])
        model_id = '{}_{}_{}_Dr_{}_{}'.format(
            self.PRE, 
            self.model.__name__, 
            '['+','.join([str(h) for h in self.model_params['h']])+']', 
            self.model_params['d'][0], 
            model_train_params)
        model_dir = '{}/{}'.format(
            self.sys_params['LOGS_BASE_FOLDER'], 
            model_id)
        self.model_dir = model_dir
        self.ptest_f = '{}/{}ptest_{}_{}bp.csv'.format(
                self.model_dir, self.PRE, self.label, self.bp)
        # self.ptest_f = '{}/Test.csv'.format(self.model_dir)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

    def set_model(self, model, model_params):
        """Set NN model and other parameters for the model to be used
        for training.

        Parameters
        ----------
        model : torch.nn.Module
            NN model used for training.
        model_params : dict
            Dictionary of model parameters that will be used to
            initialise the model object.
        """
        self.model = model
        self.model_params = model_params

    def load_plink_files(self, chroms):
        """Function to load the required plink files and phenotype-covariate
        file for the different chromosomes, before the start of training
        to speed up the data creation process.

        Parameters
        ----------
        chroms : list of str
            List of chromosomes to load. 
        """
        # Loading PLINK files for  (Dask Array)
        genotype_files = dict.fromkeys(chroms, None)
        for c in genotype_files.keys():
            plink_prefix = '{}/ukb_imp_chr{}_v3_cleaned_geno_hwe_maf_2'.format(
                self.sys_params['RAW_BASE_FOLDER'][int(c)>10], str(c))
            geno = read_plink1_bin(plink_prefix + '.bed')
            genotype_files[c] = remove_indel_multiallelic(geno)

        # Loading csv file of phenotypes and covariates (Pandas DF)
        phenotype_file = '{}/Variables_UKB.txt'.format(
            self.sys_params['RAW_BASE_FOLDER'][int(chroms[0])>10])
        phen_cov = pd.read_csv(phenotype_file, sep = ' ', dtype = {'ID_1':np.int})
        phen_cov = phen_cov.rename(columns={'ID_1':'iid'})
        phen_cov.index = phen_cov['iid']

        self.GENOTYPES = genotype_files
        self.PHEN_COV = phen_cov

    def load_cov_encodings(self, gene):
        """Load precomputed covariate encodings to speed up training.
        The encodings are duplicated and expanded to support "group 
        training".

        Returns
        -------
        tuple
            Tuple of 2 ndarrays containing the training set and testing
            set covariate encodings. 
        """
        with open(self.sys_params['COV_ENC_PATH'], 'rb') as f:
            cov_enc = pickle.load(f)
        cov_enc = np.load(self.sys_params['COV_ENC_PATH'])
        ce_train = np.expand_dims(cov_enc['train_enc'], 1)
        ce_train = np.repeat(ce_train, 10, 1)
        ce_test = np.expand_dims(cov_enc['test_enc'], 1)
        ce_test = np.repeat(ce_test, 10, 1)
        return (ce_train, ce_test)
    
    def gen_data(self, gene, chrom):
        """Generate combined data file for the gene list. Invokes 
        load_data function from dataset_utils to get the data tuple. 
        The returned data will be a combination of all snps and 
        covariates corresponding to each gene in the list. It does not 
        return individual data files for each gene in the list.

        Parameters
        ----------
        gene : list
            List of genes to generate combined data for.  
        chrom : list
            List of chromosomes (str type, not int) corresponding to the
            gene list.

        Returns
        -------
        tuple
            Data tuple containing the following in the respective
            indices:
            0 - Balanced training data (ndarray)
            1 - Balanced training labels (ndarray)
            2 - Balanced testing data (ndarray)
            3-  Balanced testing label (ndarray)
            4 - Class weights based on training labels (ndarray)
            5 - Class weights based on testing labels (ndarray)
            6 - Names of each column in the data arrays (list)
            7 - Number of SNPs in the data arrays (int)
        """
        data_arrays = load_data(
            self.GENOTYPES, self.PHEN_COV, gene, chrom, self.label, self.bp, 
            self.sys_params['LOGS_BASE_FOLDER'], self.sys_params, self.covs,
            self.OVER_COEFF, self.BALANCE, self.SNP_THRESH)
        
        X, y, X_test, y_test, class_weights, data_cols, num_snps = data_arrays
        self.covs = data_cols[num_snps:]

        #######################
        # ONLY FOR GroupTrain #
        #######################
        if len(X.shape) == 3:
            X_ = X
            y_ = y
            X_test_ = X_test
            y_test_ = y_test
        else:
            X_, y_, X_test_, y_test_ = group_data_prep(
                X, y, X_test, y_test, self.model_params['grp_size'], self.covs, 
                train_oversample=self.train_oversample, 
                test_oversample=self.test_oversample)
        cw_train = compute_class_weight(class_weight='balanced', 
            classes=np.unique(y_), y=y_)
        cw_test = compute_class_weight(class_weight='balanced', 
            classes=np.unique(y_test_), y=y_test_)

        data_tuple = (X_, y_, X_test_, y_test_, cw_train, cw_test, data_cols, 
            num_snps)

        return data_tuple

    def permloop(self, genes, perms):
        """Invoke training and permutation test for all genes passed to 
        it. Each gene (or set of genes) is invoked parallely using
        multiprocessing.Pool and trained using the corresponding set of
        in the Experiment class' 'GPU_LIST' parameter.

        Parameters
        ----------
        genes : dict
            Dictionary containing all the genes. Structure:
            {
                'chrom':[[]],
                'names':[[]]
            }
            For single genes, the 'chrom' or 'names' fields should be a list 
            containing the individual chromosomes or gene names as singleton 
            lists. For pairs of genes, they should be a list containing 
            lists representing every pair.
        perms : int
            Number of permutations for the permnutation test.
        """
        m = mp.Manager()
        lock = m.Lock()
        perm_batch_size = self.perm_batch_size
        func_args = []
        num_genes = len(genes['names'])
        cnt = 0
        self.perms = perms

        for gene_num, gene in enumerate(genes['names']):
            chrom = genes['chrom'][gene_num]
            # Skip non-autosomal chromosomes
            try: 
                tmp = int(chrom[0])
            except:                    
                continue

            # Load PLINK files if not in memory
            # if chrom[0] not in self.GENOTYPES.keys():
            #     self.load_plink_files(chrom)

            # Load PLINK files if not in memory
            if chrom[0] not in self.GENOTYPES.keys():
                self.GENOTYPES.update(dict({str(chrom[0]):None}))

            if cnt < len(self.GPU_LIST):
                print('QUEUEING {:<10} FOR TRAINING {}'.format(
                    '_'.join(gene), cnt))
                # devices = [self.GPU_LIST[cnt],]
                devices = [self.GPU_LIST[cnt],]
                fa = ([chrom, gene, devices, lock, True])
                func_args.append(fa)
                cnt+=1
            
            if cnt == len(self.GPU_LIST) or gene_num == num_genes-1:
                
                with mp.get_context('spawn').Pool(len(self.GPU_LIST)) as pool:
                    pool.starmap(self.train_gene, func_args)
                    pool.close()
                
                cnt = 0 
                func_args = []                

    def train_gene(self, chrom, gene, devices, lock, log=True):
        """Setup and run the NN training for a given gene (or combined
        set of genes). It also invokes the permutation test after training.

        Parameters
        ----------
        chrom : list
            List of chromosomes (type str) of the genes to 
            include in the dataset. For a single gene pass as a
            singleton list with the chromosome (type str).
        gene : list
            List of genes to include in the dataset. For a single gene 
            pass as a singleton list with the gene.
        devices : list
            List of GPUs to be used for training the model. The format
            of each element in the list should be a valid argument for 
            torch.device().
        lock : multiprocessing.Manager.Lock
            Lock object to prevent issues with concurrent access to the
            log files.
        log : bool, optional
            Controls if training metrics and model should be saved or
            not, by default True
        """
        
        optimiser = self.hyperparam_dict['optimiser']
        lr = self.hyperparam_dict['lr']
        batch = self.hyperparam_dict['batch']
        epochs = self.hyperparam_dict['epochs']
        gname = '_'.join(gene)
        cnum = '_'.join([str(c) for c in chrom])
        try:
            data_tuple = self.gen_data(gene, [str(c) for c in chrom])
            
            # Load the data
            if data_tuple is None:
                with open(self.NONE_DATA_FILE, 'a') as f:
                    f.write('{} has 0 SNPs!!\n'.format(gname))
                return
            
            X, y, X_test, y_test, cw, cw_test, data_cols, num_snps = data_tuple
            X = X[:, :, num_snps:]
            X_test = X_test[:, :, num_snps:]

            Xw = X
            Xw_test = X_test
                
            gn = gname
            num_snps = 0
            nsnps = num_snps

            print('{:20} Group train data: {}'.format(gn, Xw.shape))
            print('{:20} Group test data: {}'.format(gn, Xw_test.shape))
            print('{:20} Class weights: {} {}'.format(gn, cw, cw_test))

            # Model Parameters
            model_dict = {}
            model_dict['model_name'] = '{}_{}'.format(str(nsnps), gn)
            self.model_params['inp'] = Xw.shape[2]
            self.model_params['enc'] = 8
            model_dict['model_type'] = self.model
            model_dict['model_args'] = self.model_params
        

            # Optimiser Parameters
            optim_dict = {}
            optim_dict['LR'] = lr 
            optim_dict['damping'] = 1
            optim_dict['class_weights'] = cw
            optim_dict['optim'] = optimiser
            optim_dict['use_scheduler'] = False

            # Training Parameters
            train_dict = {}
            train_dict['batch_size'] = batch
            train_dict['epochs'] = epochs

            # Create all folders needed for saving training information
            if log:
                gene_dir = '{}/{}'.format(self.model_dir, gn)
                train_dict['log'] = gene_dir
                if not os.path.isdir(gene_dir):
                    os.mkdir(gene_dir)
            else:
                train_dict['log'] = None

            # If saved model exists, then skip retraining
            if not os.path.isfile('{}/{}_{}.pt'.format(
                gene_dir, model_dict['model_name'], 'Ep{}'.format(epochs-1))):
                
                with open(self.TRAIN_FILE, 'a') as f:
                    f.write('{:20} Training start\n'.format(gname))

                # Train the NN on the gene/group of genes
                start = datetime.datetime.now()
                train(Xw, y, Xw_test, y_test, nsnps, model_dict, optim_dict,
                    train_dict, devices)
                end = datetime.datetime.now()
                
                with open(self.TRAIN_FILE, 'a') as f:
                    f.write('{:20} Training time: {}\n'.format(
                        gn, end-start))
                
            # Make training plots
            create_train_plots(gene_dir, ['acc'], suffix='acc', sweight=0.0)

            # Start the permutation test
            self.start_ptest(cnum, gn, self.perms, 
                (Xw_test, y_test, cw_test, data_cols, nsnps),
                devices[0], lock)

        except:
            with open(self.TRAIN_ERR_FILE, 'a') as errf:
                errf.write(gname + '\n')
                errf.write('='*20 + '\n')
                errf.write(traceback.format_exc() + '\n\n')

    def start_ptest(self, chrom, gene, perms, data_tuple, device, lock):
        """Function to setup and run the permutation test for a given gene or 
        set of genes. Must be invoked after 'train_gene', because it will
        search and load a pretrained model to be used for the test.
        
        Parameters
        ----------
        chrom : str
            Chromosome or set of chromosomes (combined using '_'.join), 
            to run the permutation test for
        gene : str
            Gene or set of genes (combined using '_'.join), to run the
            permutation test for
        perms: int
            Number of permutations
        data_tuple : tuple
            Tuple containing the data files, Tuple indices are the same
            as that returned by Experiment class' 'gen_data' method or 
            'load_data' in dataset_utils.
        device : int, str
            Device to be used for the permutation test. Must be a valid
            argument to torch.device()
        lock : multiprocessing.Manager.Lock
            Lock object to prevent issues with concurrent access to the
            log files.
        """
        try:
            ptest_fname = self.ptest_f
            if os.path.isfile(ptest_fname):
                perm_df = pd.read_csv(ptest_fname)
                perm_df.sort_values(['Gene', 'Perms'], inplace=True, 
                    ascending=False)
                perm_df.drop_duplicates(['Gene'], inplace=True)
            else:
                cols = ['Gene', 'Chrom', 'Type', 'SNPs', 'Perms', 'F1', 'P_F1', 
                    'Prec','P_Prec', 'Rec', 'P_Rec', 'Acc', 'P_Acc', 'MCC', 
                    'P_MCC', 'Loss', 'P_Loss', 'Time']
                perm_df = pd.DataFrame(columns=cols)
            perm_df.set_index('Gene', drop=False, inplace=True)
            perm = int(perms)

            if gene in perm_df.index.values:
                prev_p = float(perm_df.loc[gene]['P_Acc'])
                if perm <= int(perm_df.loc[gene]['Perms']):
                    return
                if prev_p > 4*(10/perm):
                    return

            # Load the data
            X_test, y_test, cw_test, data_cols, num_snps = data_tuple
            
            # Permutation test parameters
            ptest_dict = {}
            ptest_dict['cross_val'] = False
            ptest_dict['num_folds'] = 10
            ptest_dict['perm_method'] = case_control_permutations
            # ptest_dict['perm_method'] = random_permutations

            model_path = '{}/{}/{}_{}.pt'.format(
                self.model_dir, gene, num_snps, gene)
            
            lock.acquire()
            with open(self.PERM_FILE, 'a') as f:
                f.write('{} : {} perm\n'.format(gene, perm))
            lock.release()

            # Perform permutation test for the gene 
            ptest_dict['num_perm'] = perm
            start = datetime.datetime.now()
            p_val, conf_mat, f1, prec, rec, acc, mcc, loss = modified_permutation_test(
                model_path, None, None, X_test, y_test, num_snps, ptest_dict, device, 
                cw_test, self.perm_batch_size)
            end = datetime.datetime.now()
            print('{:20} P-Test time: {}'.format(gene, end-start))

            # Log ptest parameters for gene 
            metric_logs = '{}/{}/ptest_metrics.npz'.format(
                self.model_dir, gene)
            np.savez(metric_logs, conf_mat=conf_mat, f1=f1, prec=prec, rec=rec, 
                acc=acc, mcc=mcc, loss=loss)
            
            lock.acquire()
            
            # If files exists, read data first to include new genes that 
            # might have completed parallely in a different process
            if os.path.isfile(ptest_fname):
                perm_df = pd.read_csv(ptest_fname)
                perm_df.set_index('Gene', drop=False, inplace=True)
            
            csv_row = {
                'Gene':[gene,],
                'Chrom':[chrom,],
                'Type':[self.gene_type,],
                'SNPs':[num_snps,],
                'Perms':[perm,],
                'F1':[round(f1[0], 4),],
                'P_F1':[p_val['p_f1'],],
                'Prec':[round(prec[0], 4),],
                'P_Prec':[p_val['p_prec'],],
                'Rec':[round(rec[0], 4),],
                'P_Rec':[p_val['p_rec'],],
                'Acc':[round(acc[0], 4),],
                'P_Acc':[p_val['p_acc'],],
                'MCC':[round(mcc[0], 4),],
                'P_MCC':[p_val['p_mcc'],],
                'Loss':[round(loss[0], 4),],
                'P_Loss':[p_val['p_loss'],],
                'Time':[str(end-start),]
            }
            tdf = pd.DataFrame.from_dict(csv_row)
            tdf.set_index('Gene', drop=False, inplace=True)
            if gene in perm_df.Gene.values:
                perm_df.update(tdf)
            else:
                perm_df = perm_df.append(tdf, verify_integrity=True)
            
            # Write data back to file so all parallel processes have
            # consistent data
            perm_df.to_csv(ptest_fname, index=False)
            lock.release()
        except:
            with open(self.PERM_ERR_FILE, 'a') as errf:
                errf.write(gene + '\n')
                errf.write('='*20 + '\n')
                errf.write(traceback.format_exc() + '\n\n')
