# coding: utf-8
import datetime
import os
import traceback
from typing import Union
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
import torch.nn as nn

from GWANN.dataset_utils import PGEN2Pandas, load_data
from GWANN.train_utils import create_train_plots, train


class Experiment:
    def __init__(self, prefix:str, label:str, params_base:str, buffer:int, 
                 model:nn.Module, model_dict:dict, hp_dict:dict, 
                 gpu_list:list):

        # Experiment descriptive parameters
        self.prefix = prefix
        self.label = label
        self.buffer = buffer
        self.params_base = params_base

        # Base parameter YAML files
        self.sys_params = None
        self.covs = None
        self.__set_params__()

        # Path parameters
        self.RUN_FOLDER = ''
        self.PERM_FILE = ''
        self.PERM_ERR_FILE = ''
        self.TRAIN_FILE = ''
        self.TRAIN_ERR_FILE = ''
        self.NONE_DATA_FILE = ''
        
        # NN model and permutation test specific parameters
        self.model = model
        self.model_params = model_dict
        self.hyperparam_dict = hp_dict
        self.model_dir = ''
        self.summary_f = ''
        self.perms = 0
        self.perm_batch_size = 1024

        self.__set_paths__()

        # Dataset generation parameters
        self.SNP_THRESH = 10000
        self.pg2pd = None
        self.phen_cov = None
        
        # Training parameters
        self.GPU_LIST = gpu_list
        self.gene_type = ''
    
    def __set_params__(self):
        """Load the YAML file containing system specific paths and 
        parameters.
        """
        sysf = f'{self.params_base}/params_{self.label}.yaml'
        with open(sysf, 'r') as params_file:
            self.sys_params = yaml.load(params_file, Loader=yaml.FullLoader) 
        covsf = f'{self.params_base}/covs_{self.label}.yaml'
        with open(covsf, 'r') as covs_file:
            self.covs = yaml.load(covs_file, Loader=yaml.FullLoader)['COVARIATES']
        
    def __set_paths__(self):
        """Function to set all file and folder paths, later used by
        other functions.
        """
        self.RUN_FOLDER = '{}/{}'.format(self.sys_params['RUNS_BASE_FOLDER'], self.prefix)
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
            self.prefix, 
            self.model.__name__, 
            '['+','.join([str(h) for h in self.model_params['h']])+']', 
            self.model_params['d'][0], 
            model_train_params)
        model_dir = '{}/{}'.format(
            self.sys_params['LOGS_BASE_FOLDER'], 
            model_id)
        self.model_dir = model_dir
        self.summary_f = '{}/{}_summary_{}_{}bp.csv'.format(
                self.model_dir, self.prefix, self.label, self.buffer)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
    
    def __set_genotypes_and_covariates__(self, chrom):
        pgen_prefix = f'{self.sys_params["RAW_BASE_FOLDER"][chrom]}/UKB_chr{chrom}'
        train_ids = pd.read_csv(self.sys_params["TRAIN_IDS_PATH"], 
                                dtype={'iid':str})['iid'].to_list()
        test_ids = pd.read_csv(self.sys_params["TEST_IDS_PATH"],
                            dtype={'iid':str})['iid'].to_list()
        self.pg2pd = PGEN2Pandas(pgen_prefix, sample_subset=train_ids+test_ids)
        
        self.phen_cov = pd.read_csv(self.sys_params['PHEN_COV_PATH'], 
                            sep=' ', dtype={'ID_1':str}, comment='#')
        self.phen_cov = self.phen_cov.rename(columns={'ID_1':'iid'})
        self.phen_cov.index = self.phen_cov['iid']

    def __gen_data__(self, gene:str, chrom:str, start:int, end:int) -> tuple:
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
        start : int
            Start position of gene on chromsome.
        end : int
            End position of gene on chromsome.
            
        Returns
        -------
        tuple
            Data tuple containing the following in the respective
            indices:
            0 - Grouped training data (ndarray)
            1 - Grouped training labels (ndarray)
            2 - Grouped testing data (ndarray)
            3-  Grouped testing label (ndarray)
            4 - Class weights based on training labels (ndarray)
            5 - Names of each column in the data arrays (list)
            6 - Number of SNPs in the data arrays (int)
        """

        data = load_data(pg2pd=self.pg2pd, phen_cov=self.phen_cov, gene=gene, 
                         chrom=chrom, start=start, end=end, buffer=self.buffer, 
                         label=self.label, sys_params=self.sys_params, 
                         covs=self.covs, SNP_thresh=self.SNP_THRESH, 
                         only_covs=True, lock=None)
        
        return data

    def parallel_run(self, genes:dict):
        """Invoke training and permutation test for all genes passed to 
        it. Each gene (or set of genes) is invoked parallely using
        multiprocessing.Pool and trained using the corresponding set of
        in the Experiment class' 'GPU_LIST' parameter.

        Parameters
        ----------
        genes : dict
            Dictionary containing all the genes. Structure:
                {'chrom':list, 'names':list, 'start':list, 'end':list}
        """
        
        m = mp.Manager()
        lock = m.Lock()
        func_args = []
        num_genes = len(genes['names'])
        cnt = 0
        for gene_num, gene in enumerate(genes['names']):
            chrom = genes['chrom'][gene_num]
            start = genes['start'][gene_num]
            end = genes['end'][gene_num]
            if cnt < len(self.GPU_LIST):
                print(f'QUEUEING {gene} FOR TRAINING {cnt}')
                device = self.GPU_LIST[cnt]
                func_args.append((chrom, gene, start, end, device, lock, True))
                cnt+=1
            
            if cnt == len(self.GPU_LIST) or gene_num == num_genes-1:
                
                with mp.get_context('spawn').Pool(len(self.GPU_LIST)) as pool:
                    pool.starmap(self.train_gene, func_args)
                    pool.close()
                    pool.join()
                cnt = 0 
                func_args = []                

    def train_gene(self, chrom:str, gene:str, start:int, end:int, 
                   device:Union[str, int], lock:mp.Lock, log:bool=True) -> None:
        """Setup and run the NN training for a given gene. It also
        invokes the permutation test after training.

        Parameters
        ----------
        chrom : list
            List of chromosomes (type str) of the genes to 
            include in the dataset. For a single gene pass as a
            singleton list with the chromosome (type str).
        gene : list
            List of genes to include in the dataset. For a single gene 
            pass as a singleton list with the gene.
        start : int
            Start position of gene.
        end : int
            End position of gene.
        device : int, str
            GPU to be used for training the model. The format
            of each element in the list should be a valid argument for 
            torch.device().
        lock : mp.Lock
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
        
        try:
            # Load the data
            data_tuple = self.__gen_data__(gene, chrom, start, end)
            if data_tuple is None:
                with open(self.NONE_DATA_FILE, 'a') as f:
                    f.write(f'{gene} has no data file!!\n')
                return
            
            X, y, X_test, y_test, cw, data_cols, num_snps = data_tuple
            assert num_snps == 0            

            print(f'{gene:20} Group train data: {X.shape}')
            print(f'{gene:20} Group test data: {X_test.shape}')
            print(f'{gene:20} Class weights: {cw}')

            # Model Parameters
            model_dict = {}
            model_dict['model_name'] = f'{num_snps}_{gene}'
            self.model_params['inp'] = X.shape[2]
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
                gene_dir = '{}/{}'.format(self.model_dir, gene)
                train_dict['log'] = gene_dir
                if not os.path.isdir(gene_dir):
                    os.mkdir(gene_dir)
            else:
                train_dict['log'] = None

            # If saved model exists, then skip retraining
            if not os.path.isfile(f'{gene_dir}/{model_dict["model_name"]}_Ep{epochs-1}.pt'):
                
                with open(self.TRAIN_FILE, 'a') as f:
                    f.write(f'{gene:20} Training start\n')

                # Train the NN on the gene
                st = datetime.datetime.now()
                best_acc, best_loss = train(X=X, y=y, X_test=X_test, y_test=y_test, 
                                model_dict=model_dict, optim_dict=optim_dict, 
                                train_dict=train_dict, device=device)
                et = datetime.datetime.now()
                
                with open(self.TRAIN_FILE, 'a') as f:
                    f.write(f'{gene:20} Training time: {et-st}\n')
                
                self.__write_gene_summary_row__(gene=gene, chrom=chrom, 
                                    num_snps=num_snps, acc=best_acc, 
                                    loss=best_loss, time_taken=str(et-st), 
                                    lock=lock)
                
            # Make training plots
            create_train_plots(gene_dir, ['acc'], suffix='acc', sweight=0.0)
        except:
            with open(self.TRAIN_ERR_FILE, 'a') as errf:
                errf.write(gene + '\n')
                errf.write('='*20 + '\n')
                errf.write(traceback.format_exc() + '\n\n')

    def __write_gene_summary_row__(self, gene:str, chrom:str, num_snps:int, acc:float, 
                               loss:float, time_taken:str, lock:mp.Lock) -> None:
        """_summary_

        Parameters
        ----------
        gene : str
            _description_
        chrom : str
            _description_
        num_snps : int
            _description_
        acc : float
            _description_
        loss : float
            _description_
        time_taken : str
            _description_
        lock : mp.Lock
            _description_
        """
        lock.acquire()
        
        # If files exists, read data first to include new genes that 
        # might have completed parallely in a different process
        if os.path.isfile(self.summary_f):
            summ_df = pd.read_csv(self.summary_f)
            summ_df.set_index('Gene', drop=False, inplace=True)
        else:
            summ_df = pd.DataFrame(columns=['Gene', 'Chrom', 'Type', 'SNPs', 
                                            'Acc', 'Loss', 'Time'])

        csv_row = {
            'Gene':[gene,],
            'Chrom':[chrom,],
            'Type':[self.gene_type,],
            'SNPs':[num_snps,],
            'Acc':[acc,],
            'Loss':[loss,],
            'Time':[time_taken,]
        }
        tdf = pd.DataFrame.from_dict(csv_row)
        tdf.set_index('Gene', drop=False, inplace=True)
        if gene in summ_df.Gene.values:
            summ_df.update(tdf)
        else:
            summ_df = summ_df.append(tdf, verify_integrity=True)
        
        # Write data back to file so all parallel processes have
        # consistent data
        summ_df.to_csv(self.summary_f, index=False)
        lock.release()
