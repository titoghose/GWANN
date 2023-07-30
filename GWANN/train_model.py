# coding: utf-8
import csv
import datetime
import os
import traceback
import warnings
from typing import Optional, Union
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import torch

warnings.filterwarnings('ignore')

import multiprocessing as mp

import numpy as np
import pandas as pd
import torch.nn as nn
import yaml
import shap
import time

from GWANN.dataset_utils import PGEN2Pandas, load_data, load_win_data
from GWANN.train_utils import create_train_plots, train
from GWANN.models import Diff, Identity

class FullModel(torch.nn.Module):
    """
    """
    def __init__(self, gene_model, cov_model):
        super(FullModel, self).__init__()

        self.gene_model = gene_model
        self.cov_model = cov_model
        self.cov_model.end_model.linears[-1] = Identity()
        
    def forward(self, x):
        snps_enc = self.gene_model.snp_enc(torch.transpose(x[:, :, :self.gene_model.num_snps], 1, 2))
        snps_pooled = torch.squeeze(self.gene_model.pool_ind(snps_enc), dim=-1)
        self.att_out = self.gene_model.att_mask(snps_pooled)
        self.att_out.requires_grad_(True)
        features_pooled = self.gene_model.pool_features(
            torch.unsqueeze(self.att_out, dim=1))
        snps_out = self.gene_model.snps_model(torch.squeeze(features_pooled, dim=1))
        
        cov_out = self.cov_model(x[:, :, self.gene_model.num_snps:])
        data_vec = torch.cat((snps_out, cov_out), dim=-1)
        raw_out = self.gene_model.end_model(data_vec)
        
        return raw_out

class Experiment:
    def __init__(self, prefix:str, label:str, params_base:str, buffer:int, 
                 model:nn.Module, model_dict:dict, hp_dict:dict, 
                 gpu_list:list, only_covs:bool, grp_size:int=10):

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
        self.grp_size = grp_size
        self.model_dir = ''
        self.summary_f = ''
        self.perms = 0
        self.perm_batch_size = 1024

        self.__set_paths__()

        # Dataset generation parameters
        self.SNP_THRESH = 10000
        self.pg2pd = None
        self.phen_cov = None
        self.only_covs = only_covs
        self.cov_encodings = self.__load_cov_encodings__()

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
        self.summary_f = '{}/{}_{}bp_summary.csv'.format(
                self.model_dir, self.prefix, self.buffer)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
    
    def __set_genotypes_and_covariates__(self, chrom:str) -> None:
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

    def __gen_data__(self, gene_dict:dict, only_covs:bool=False) -> tuple:
        """Generate combined data file for the gene list. Invokes 
        load_data function from dataset_utils to get the data tuple. 
        The returned data will be a combination of all snps and 
        covariates corresponding to each gene in the list. It does not 
        return individual data files for each gene in the list.

        Parameters
        ----------
        gene_dict : dict
            Dict of gene info. Keys 'name' and 'chrom' are required and
            either one of (i) 'start and 'end' or (ii) 'win'.
        only_covs : bool, optional
            Load data with only covariate columns, by default False.
        
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
        gene = gene_dict['gene']
        chrom = gene_dict['chrom']
        start = gene_dict['start'] if 'start' in gene_dict else None
        end = gene_dict['end'] if 'end' in gene_dict else None
        win = gene_dict['win'] if 'win' in gene_dict else None

        if win is None:
            data = load_data(pg2pd=self.pg2pd, phen_cov=self.phen_cov, gene=gene, 
                         chrom=chrom, start=start, end=end, buffer=self.buffer, 
                         label=self.label, sys_params=self.sys_params, 
                         covs=self.covs, SNP_thresh=self.SNP_THRESH, 
                         only_covs=only_covs, lock=None)
        else:
            data = load_win_data(gene=gene, win=win, chrom=chrom, 
                        buffer=self.buffer, label=self.label, 
                        sys_params=self.sys_params, covs=self.covs, 
                        only_covs=only_covs)
            
        return data

    def __load_cov_encodings__(self) -> Optional[dict]:
        """Load covariate encodings saved from the covariate model.

        Returns
        -------
        Optional[dict]
            If the pipeline is for the covariate model, return None. If
            not, load saved covariate encodings and return dictionary of
            train and test encodings.
        """
        if self.only_covs:
            return None
        else:
            cov_enc = np.load(self.sys_params['COV_ENC_PATH'])
        
            ce_train = np.expand_dims(cov_enc['train_enc'], 1)
            ce_train = np.repeat(ce_train, self.grp_size, 1)
            ce_test = np.expand_dims(cov_enc['test_enc'], 1)
            ce_test = np.repeat(ce_test, self.grp_size, 1)
            return {'train': ce_train, 'test': ce_test}

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
        num_genes = len(genes['gene'])
        cnt = 0
        shared_gpu_stack = m.list(self.GPU_LIST)
        # for gene_num, gene in enumerate(genes['gene']):
        #     gdict = {k:genes[k][gene_num] for k in genes.keys()}
        #     if cnt < len(self.GPU_LIST):
        #         print(f'QUEUEING {gene} FOR TRAINING {cnt}')
        #         device = self.GPU_LIST[cnt]
        #         func_args.append((shared_gpu_idx, gdict, device, lock, True))
        #         cnt+=1
            
        #     if cnt == len(self.GPU_LIST) or gene_num == num_genes-1:
                
        #         with mp.get_context('spawn').Pool(len(self.GPU_LIST)) as pool:
        #             pool.starmap(self.train_gene, func_args)
        #             pool.close()
        #             pool.join()
        #         cnt = 0 
        #         func_args = []                

        for gene_num, gene in enumerate(genes['gene']):
            gdict = {k:genes[k][gene_num] for k in genes.keys()}
            # print(f'QUEUEING {gene} FOR TRAINING {cnt}')
            device = self.GPU_LIST[cnt]
            func_args.append((shared_gpu_stack, gdict, device, lock, True))
            
        with mp.get_context('spawn').Pool(len(self.GPU_LIST)) as pool:
            pool.starmap_async(self.train_gene, func_args, chunksize=1)
            pool.close()
            pool.join()
        
    def train_gene(self, shared_gpu_stack:list, gene_dict:dict, device:Union[str, int], 
                   lock:mp.Lock, log:bool=True) -> None:
        """Setup and run the NN training for a given gene. It also
        invokes the permutation test after training.

        Parameters
        ----------
        gene_dict : dict
            Dict of gene info. Keys 'name' and 'chrom' are required and
            either one of (i) 'start and 'end' or (ii) 'win'.
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
        with lock:
            device = int(shared_gpu_stack.pop(0))

        gene = gene_dict['gene']
        chrom = gene_dict['chrom']
        if 'win' in gene_dict:
            gene = f'{gene}_{gene_dict["win"]}'

        print(f'Running {gene} on gpu: {device}')

        try:
            # print(f'Hello, {gene} is running on {device}')
            # tensor = torch.from_numpy(np.ones((1000, 1000))).to(device)
            # time.sleep(np.random.randint(1, 5))
            # Load the data
            data_tuple = self.__gen_data__(gene_dict=gene_dict, 
                                           only_covs=self.only_covs)
            if data_tuple is None:
                with open(self.NONE_DATA_FILE, 'a') as f:
                    f.write(f'{gene} has no data file!!\n')
                return
            
            X, y, X_test, y_test, cw, data_cols, num_snps = data_tuple
            if self.only_covs:
                assert num_snps == 0
                assert len(data_cols) == len(self.covs)

            if not self.only_covs:
                X = np.concatenate(
                    (X[:, :, :num_snps], self.cov_encodings['train']), 
                    axis=-1)
                X_test = np.concatenate(
                    (X_test[:, :, :num_snps], self.cov_encodings['test']), 
                    axis=-1)

            print(f'{gene:20} Group train data: {X.shape}')
            print(f'{gene:20} Group test data: {X_test.shape}')
            print(f'{gene:20} Class weights: {cw}')

            # Model Parameters
            model_dict = {}
            model_dict['model_name'] = f'{num_snps}_{gene}'
            self.model_params['inp'] = len(self.covs) if self.only_covs else num_snps
            model_dict['model_type'] = self.model
            model_dict['model_args'] = self.model_params
        
            # Optimiser Parameters
            optim_dict = {}
            optim_dict['LR'] = self.hyperparam_dict['lr'] 
            optim_dict['damping'] = 1
            optim_dict['class_weights'] = cw
            optim_dict['optim'] = self.hyperparam_dict['optimiser']
            optim_dict['use_scheduler'] = False

            # Training Parameters
            train_dict = {}
            train_dict['batch_size'] = self.hyperparam_dict['batch']
            train_dict['epochs'] = self.hyperparam_dict['epochs']
            train_dict['early_stopping'] = self.hyperparam_dict['early_stopping']

            # Create all folders needed for saving training information
            if log:
                gene_dir = '{}/{}'.format(self.model_dir, gene)
                train_dict['log'] = gene_dir
                if not os.path.isdir(gene_dir):
                    os.mkdir(gene_dir)
            else:
                train_dict['log'] = None

            # If saved model exists, then skip retraining
            model_ckpt = f'{gene_dir}/{model_dict["model_name"]}.pt'
            if not os.path.isfile(model_ckpt):
                
                with open(self.TRAIN_FILE, 'a') as f:
                    f.write(f'{gene:20} Training start\n')

                # Train the NN on the gene
                st = datetime.datetime.now()
                best_epoch, best_acc, best_loss = train(X=X, y=y, 
                                X_test=X_test, y_test=y_test, 
                                model_dict=model_dict, optim_dict=optim_dict, 
                                train_dict=train_dict, device=device)
                et = datetime.datetime.now()
                
                with open(self.TRAIN_FILE, 'a') as f:
                    f.write(f'{gene:20} Training time: {et-st}\n')
                
                self.__write_gene_summary_row__(gene=gene, chrom=chrom, 
                                    num_snps=num_snps, epoch=best_epoch, 
                                    acc=best_acc, loss=best_loss, 
                                    time_taken=str(et-st), lock=lock)
                
            # Make training plots
            create_train_plots(gene_dir, ['acc'], suffix='acc', sweight=0.0)
            create_train_plots(gene_dir, ['loss'], suffix='loss', sweight=0.0)

        except:
            with open(self.TRAIN_ERR_FILE, 'a') as errf:
                errf.write(gene + '\n')
                errf.write('='*20 + '\n')
                errf.write(traceback.format_exc() + '\n\n')
        finally:
            with lock:
                shared_gpu_stack.append(device)

    def __write_gene_summary_row__(self, gene:str, chrom:str, num_snps:int, epoch:int, 
                                   acc:float, loss:float, time_taken:str, lock:mp.Lock) -> None:
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
        try:
            lock.acquire()
            header = ['Gene', 'Chrom', 'Type', 'SNPs', 'Epoch', 'Acc', 'Loss', 'Time']
            data_info = [gene, chrom, self.gene_type, num_snps, epoch, acc, loss, time_taken]
            if os.path.isfile(self.summary_f):
                rows = [data_info]
            else:
                rows = [header, data_info]
        
            with open(self.summary_f, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        
        except Exception as e:
            raise e
        finally:
            lock.release()

    def calculate_shap(self, gene_dict:dict, device:str) -> Figure:
        
        gene = gene_dict['gene']
        if 'win' in gene_dict:
            gene = f'{gene}_{gene_dict["win"]}'
        data_tuple = self.__gen_data__(gene_dict=gene_dict, 
                                       only_covs=self.only_covs)
        X, y, X_test, y_test, cw, data_cols, num_snps = data_tuple
        if self.only_covs:
            assert num_snps == 0
            assert len(data_cols) == len(self.covs)

        # if not self.only_covs:
        #     X = np.concatenate(
        #         (X[:, :, :num_snps], self.cov_encodings['train']), 
        #         axis=-1)
        #     X_test = np.concatenate(
        #         (X_test[:, :, :num_snps], self.cov_encodings['test']), 
        #         axis=-1)

        print(f'{gene:20} Group train data: {X.shape}')
        print(f'{gene:20} Group test data: {X_test.shape}')
        print(f'{gene:20} Class weights: {cw}')

        model_name = f'{num_snps}_{gene}'
        gene_dir = '{}/{}'.format(self.model_dir, gene)
        model_path = f'{gene_dir}/{model_name}.pt'

        model = torch.load(model_path, map_location=torch.device(device))
        if not self.only_covs:
            cov_model_path = f'/home/upamanyu/GWANN/Code_AD/NN_Logs/{self.prefix.replace("Chr", "Cov")}_GroupAttention_[128,64,16]_Dr_0.3_LR:0.0001_BS:256_Optim:adam/BCR/0_BCR.pt'
            cov_model = torch.load(cov_model_path, map_location=torch.device(device))
            model = FullModel(model, cov_model).to(device)
        
        # model = torch.nn.Sequential(
        #     model,
        #     Diff()
        # ).to(device)

        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        data = X_test[torch.randperm(len(X_test))[:1000]]
        e = shap.DeepExplainer(model, X[torch.randperm(len(X))[:1000]])
        shap_vals = e.shap_values(data)[0]
        shap_vals = np.reshape(shap_vals, (shap_vals.shape[0]*shap_vals.shape[1], -1))

        feature_names = data_cols
        data = data.detach().cpu().numpy()
        data = np.reshape(data, (data.shape[0]*data.shape[1], -1))
        shap_plot_df = pd.DataFrame(data, columns=feature_names)
        
        fig = plt.figure()
        sort_features = True
        shap.summary_plot(shap_vals, shap_plot_df, plot_size=(7, 5), 
                max_display=20, sort=sort_features, alpha=0.5, 
                color_bar_label='')
        return fig
