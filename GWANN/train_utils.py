
# coding: utf-8
import os
import csv
import ast
from typing import Union
import tqdm
import yaml
import copy
import math
import bisect
import datetime
import traceback
import numpy as np
import pandas as pd
import itertools as it
import multiprocessing as mp
from torchviz import make_dot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
plt.rcParams['svg.fonttype'] = 'none'
from adjustText import adjust_text

from GWANN.models import *
from GWANN.models import *
from GWANN.dataset_utils import *

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from statsmodels.stats.proportion import proportion_confint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load paramaters
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, metric, model, inc=True):
        
        score = metric if inc else -metric
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        # if self.verbose:
        #     self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, self.path)

class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side 
    from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, 
                            vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        if clip:
            return value
        else:
            return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

class GWASDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data (seq_len, batch, input_dim), label for index 
        return (self.data[idx], self.labels[idx])

class FastTensorDataLoader:
    """A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """Initialize a FastTensorDataLoader.
        
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 1:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_batches*self.batch_size:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


# Visualisation functions

def create_perm_plots(gene_dir, m_plot, save_as='svg', suffix=''):
    return

def comparitive_train_plots(logs_dir, gene, m_plot, sweight=0, save_as='svg', 
                            suffix=''):
    """Generate plots for training metrics.

        Parameters
    ----------
    logs_dir: str
        Path to base folder containing all the log files.
    gene : str
        Name of the gene.
    mplot: list of str or None
        Metrics used to generate the different plots (default=['acc',
         'loss']). If None, all metrics will be plot.
    sweight : Value ranging between 0 and 1 specifying the extent of
        smooto be applied to the plots.
    tv: list of str
        Group to generate plot for (default=['train', 'val']).
    save_as:str
        Extension for the plot (default='svg').
    suffix: str
        Suffix to add to the figure name (default='').

    """
    
    # Create plot for each metric
    fig, ax = plt.subplots()
    fig_name = '{}/Compare_Plots/{}_{}.{}'.format(logs_dir, gene, suffix, save_as)
    
    for d in os.listdir(logs_dir):
        d_path = os.path.join(logs_dir, d)
        if not os.path.isdir(d_path):
            continue
        
        gene_dir = '{}/{}'.format(d_path, gene)
        try:
            tm = np.load('{}/training_metrics.npz'.format(gene_dir))
        except FileNotFoundError:
            print(gene_dir)
            continue
        
        # Get values only for the first FOLD
        conf_mat = tm['agg_conf_mat'][0]
        loss = tm['agg_loss'][0]
        
        metrics = {'f1':[[], []], 'prec':[[], []], 'rec':[[], []], 'acc':[[], []], 
            'mcc':[[], []], 'loss':[loss[:,0], loss[:,1]]}
        for i, cm in enumerate(conf_mat):
            tf, tp, tr, ta, tm = metrics_from_conf_mat(cm[0])
            vf, vp, vr, va, vm = metrics_from_conf_mat(cm[1])
            metrics['f1'][0].append(tf)
            metrics['prec'][0].append(tp)
            metrics['rec'][0].append(tr)
            metrics['acc'][0].append(ta)
            metrics['mcc'][0].append(tm)
            metrics['f1'][1].append(vf)
            metrics['prec'][1].append(vp)
            metrics['rec'][1].append(vr)
            metrics['acc'][1].append(va)
            metrics['mcc'][1].append(vm)

        if m_plot is None:
            m_plot = metrics.keys()
        for m in m_plot:
            # Initialise graph values
            yt = smoothen(metrics[m][0], sweight)
            yv = smoothen(metrics[m][1], sweight)
            x = np.arange(len(yt))
            
            # Create and save the graph
            train_line, = ax.plot(x, yt, label=d, linestyle='-')
            ax.plot(x, yv, label=d, linestyle=':', c=train_line.get_color())
            ax.tick_params(axis='both', labelsize=8)
            
    ax.set_xlabel('Epochs')
    ax.set_ylabel('_'.join(m_plot))
    ax.legend(loc='lower right', fontsize=4)
    fig.savefig(fig_name)
    plt.close(fig)

def create_train_plots(gene_dir, m_plot, sweight=0, save_as='svg', suffix=''):
    """Generate plots for training metrics.

    Parameters
    ----------
    gene_dir: str
        Path to base folder containing all the log files.
    mplot: list of str or None
        Metrics used to generate the different plots (default=['acc',
         'loss']). If None, all metrics will be plot.
    sweight : Value ranging between 0 and 1 specifying the extent of
        smooto be applied to the plots.
    tv: list of str
        Group to generate plot for (default=['train', 'val']).
    save_as:str
        Extension for the plot (default='svg').
    suffix: str
        Suffix to add to the figure name (default='').

    """
    
    fig_name = '{}/train_plot_{}.{}'.format(gene_dir, suffix, save_as)
    
    colors = {'f1':'blue', 'prec':'green', 'rec':'violet', 'acc':'orange', 
        'mcc':'black', 'loss':'red'}
    tm = np.load('{}/training_metrics.npz'.format(gene_dir))
    
    # Get values only for the first FOLD
    conf_mat = tm['agg_conf_mat']
    loss = tm['agg_loss']
    
    metrics = {'f1':[[], []], 'prec':[[], []], 'rec':[[], []], 'acc':[[], []], 
        'mcc':[[], []], 'loss':[loss[:,0], loss[:,1]]}
    for i, cm in enumerate(conf_mat):
        tf, tp, tr, ta, tm = metrics_from_conf_mat(cm[0])
        vf, vp, vr, va, vm = metrics_from_conf_mat(cm[1])
        metrics['f1'][0].append(tf)
        metrics['prec'][0].append(tp)
        metrics['rec'][0].append(tr)
        metrics['acc'][0].append(ta)
        metrics['mcc'][0].append(tm)
        metrics['f1'][1].append(vf)
        metrics['prec'][1].append(vp)
        metrics['rec'][1].append(vr)
        metrics['acc'][1].append(va)
        metrics['mcc'][1].append(vm)

    # Create plot for each metric
    fig, ax = plt.subplots()
    
    if m_plot is None:
        m_plot = metrics.keys()
    for m in m_plot:
        # Initialise graph values
        yt = smoothen(metrics[m][0], sweight)
        yv = smoothen(metrics[m][1], sweight)
        x = np.arange(len(yt))
        
        # Create and save the graph
        ax.plot(x, yt, label=m+'_train', linestyle='-', c=colors[m])
        ax.plot(x, yv, label=m+'_test', linestyle=':', c=colors[m])
        ax.tick_params(axis='both', labelsize=8)
        
    ax.set_xlabel('Epochs')
    ax.set_ylabel('_'.join(m_plot))
    ax.legend()
    fig.savefig(fig_name)
    plt.close(fig)
              
def smoothen(data, weight):
    """Function to smoothen a data array. If any value is missing
        (represented by NaN), the value at that timestep is replaced by
        the smoothened value from the previous timestep.

    Parameters
    ----------
    data : list, ndarray
        Array containing the data points.
    weight : float
        The extent of smoothing (equivalent to a window for a running 
        average).

    Returns
    -------
    ndarray
        Smoothened data array.
    """
    smooth_data = []
    s_data = data[0]
    smooth_data.append(s_data)
    for d in data[1:]:
        if d == float('nan') or np.isnan(d):
            smooth_data.append(smooth_data[-1])    
            continue
        s_data = (weight*s_data) + (1-weight)*d
        smooth_data.append(s_data)
    return np.array(smooth_data)

def hyperparameter_plot2(log_dir, log_file, genes):
    """
    Function to generate a hyperparamter comparison plot between 
    grid and random search.

    Parameters
    ----------
    log_dir: str
        Path to the base folder containing all log files.
    log_file: str
        Specific log file containing hyperparamter search logs.
    genes: list of str
        List of genes to consider while calculating average 
        metric values.

    """
    hunits = list(map(lambda x:2**x, np.arange(10,0,-1)))
    hidden = list(it.product(hunits, hunits))
    hidden.extend(list(it.product(hunits, hunits, hunits)))
    hidden = [str(x) for x in hidden]

    tmp = pd.DataFrame(hidden, columns=['hidden'])
    tmp['sort1'] = tmp['hidden'].apply(lambda x:len(ast.literal_eval(x)))
    tmp['sort2'] = tmp['hidden'].apply(lambda x:ast.literal_eval(x)[0])
    tmp['sort3'] = tmp['hidden'].apply(lambda x:ast.literal_eval(x)[1])
    tmp['sort4'] = tmp['hidden'].apply(lambda x:ast.literal_eval(x)[-1])
    tmp.sort_values(['sort1','sort2','sort3','sort4'], inplace=True)
    
    hidden = tmp['hidden'].to_list()
    x = np.arange(len(hidden))
    drop = [0.1, 0.2, 0.3, 0.4, 0.5]
    lr = 0.01
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(30, 8), dpi=300)
    ax = ax.flatten()
    min_loss1 = 100
    min_loss2 = 100
    max_acc1 = 0
    max_acc2 = 0
    
    for grid, col in zip(['/grid', ''], ['Blues', 'Reds']):
        # Grid Search
        h_dict = {str(h):[0, 0, 0] for h in hidden}
        mat = np.zeros((len(drop)*10, len(hidden)))
        for gene in tqdm.tqdm(genes):
            fname = '{}/Genes/{}{}/{}'.format(log_dir, gene, grid, log_file)
            df = pd.read_csv(fname)
            tmp_dict = {str(h):[100, 0, 0] for h in hidden}
            for _, row in df.iterrows():
                if row['LR'] == 0.001:
                    continue

                h = str(row['hidden'])
                d = float(row['dropout'])
                cnt = h_dict[h][2]
                tmp_dict[h][0] = min(tmp_dict[h][0], row['val_loss'])
                tmp_dict[h][1] = max(tmp_dict[h][1], row['val_acc'])
                tmp_dict[h][2] += 1
                
                hi = hidden.index(h)
                di = drop.index(d)
                mat[di*10:(di+1)*10, hi] += row['val_acc']
            
            for h in hidden:
                h = str(h)
                h_dict[h][0] += tmp_dict[h][0]
                h_dict[h][1] += tmp_dict[h][1]
                h_dict[h][2] += tmp_dict[h][2]

        acc = np.array([h_dict[h][1] for h in hidden])/len(genes)
        max_acc1 = max(np.max(acc), max_acc1)
        acc = -1*(acc - (max_acc1 + 1e-4))
        acc[acc==(max_acc1 + 1e-4)] = 0
        nll = np.zeros(len(acc))
        np.log(acc, out=nll, where=(acc!=0))
        nll *= -1
        ax[0].scatter(x, nll, c=col[0].lower())
        
        mat = mat/len(genes)
        max_acc2 = max(np.max(mat), max_acc2)
        mat = -1*(mat - (max_acc2 + 1e-4))
        mat[mat==(max_acc2 + 1e-4)] = 0
        mat_nll = np.zeros(mat.shape)
        np.log(mat, out=mat_nll, where=(mat!=0))
        mat_nll *= -1
        cmap = cm.get_cmap(col)
        cmap.set_under('k', alpha=0)
        ax[1].imshow(mat_nll, cmap=cmap, origin='bottom', vmin=1e-10)
        if 'Blues' in col:
            ax[0].axvline(np.argmax(nll), c='k', linestyle=':')
            ax[1].axvline(np.argmax(nll), c='k', linestyle=':')
        
        print(max_acc1, max_acc2, hidden[np.argmax(nll)])
    
    for x in np.arange(0,10*len(drop),10):
        ax[1].axhline(x-0.5, c='k', linestyle='-', linewidth=0.5)
    
    ax[0].tick_params(labelsize=14)
    ax[1].tick_params(labelsize=14)
    ax[1].set_yticks(np.arange(5,len(drop)*10, 10)-0.5)
    ax[1].set_yticklabels([0.1, 0.2, 0.3, 0.4, 0.5])
    fig.savefig('{}/Figures/hyp_plot_grid.svg'.format(log_dir))
    plt.close()

def input_perturbation(gene, model_path, X, y, num_snps, class_weights, 
                        columns, device, folder, prefix):
    """
    Function that perturbs each input in the data vector (one at a time) 
    for a neural network model and then calculates the change in loss 
    compared to the baseline model with no perturbations. It acts as a 
    proxy for the importance of elements in the data vector. A higher 
    increase in loss corresponds to greater importance.

    Parameters
    ----------
    gene : list of str
        List of gene names.
    model_path : list of str
        List of paths of saved PyTorch models for each gene.
    X : list of Numpy ndarray
        List of data for each gene.
    y : list of Numpy ndarray
        List of training labels for each gene.
    num_snps : list of int
        Number of SNPs in each gene.  
    class_weights : list of Numpy ndarray
        List of class weights for each gene for the loss function.
    columns : List of lists
        List of column headers for the data (SNPs, age, etc.) in each 
        gene.
    device : str
        Device for inference (eg: 'cuda:0').
    folder : str
        Base folder for saving generated plots.
    prefix : str
        Prefix to add to the file name while saving the plots.
    
    """
    mat = []
    gs = []
    for i, g in enumerate(gene):
        # g = '_'.join(gene_set)
        gs.append(g)
        print('\t\t Perturbation for {}'.format(g))

        colsg = columns[i]
        modelg = torch.load(model_path[i])
        loss_fng = nn.CrossEntropyLoss(torch.tensor(class_weights[i]).float())
        Xg = X[i]
        yg = y[i]
        yg = torch.from_numpy(yg).long()

        Xg_shape = Xg.shape
        Xg = np.reshape(Xg, (Xg.shape[0], -1))
        np.random.seed(725)
        noise_vec = np.random.normal(0, 1, Xg.shape[0])
        
        accs = []
        
        for j, inp in enumerate([None, ]+colsg):
            Xg_ = Xg.copy()
            if j >= num_snps[i]:
                for k in range(-12, 0, -1):
                    Xg_[:, k] = noise_vec
            else:
                for k in range(0, num_snps[i]):
                    Xg_[:, k] = noise_vec

            Xg_ = np.reshape(Xg_, Xg_shape)
            Xg_ = torch.from_numpy(Xg_).float()

            _, conf_mat, loss = infer(Xg_, yg, modelg, loss_fng, device, 
                class_weights=class_weights[i], batch_size=1024)
            _, _, _, acc, _ = metrics_from_conf_mat(conf_mat[0])
            accs.append(acc)

        mat.append(np.array(accs)[[1, -2]]-accs[0])
        # fig, ax = plt.subplots()
        # ax.scatter(colsg, np.array(accs)[1:]-accs[0], s=4)
        # ax.axvline(num_snps[i]-0.5, linestyle=':', c='g', alpha=0.5)
        # ax.axhline(0, linestyle=':', c='r', alpha=0.5)
        # ax.set_xticklabels([])
        # fig.savefig('{}/{}_{}.svg'.format(folder, prefix, g))
    
    fig, ax = plt.subplots()
    mat = np.asarray(mat)
    mat = (mat - np.mean(mat))/np.std(mat)
    img = ax.imshow(mat.T, cmap='Reds', origin='lower', vmin=0)
    # ax.axvline(num_snps[i]-0.5, linestyle=':', c='g', alpha=0.5)
    # ax.axhline(0, linestyle=':', c='r', alpha=0.5)
    ax.set_xticks(np.arange(len(gs)))
    ax.set_xticklabels(gs, {'fontsize': 6})
    ax.set_yticks(np.arange(2))
    ax.set_yticklabels(['SNPs', 'PCs'], {'fontsize': 6})
    # ax.set_xticks(np.arange(12))
    # ax.set_xticklabels(['Sex', 'Age', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 
    #     'PC6', 'PC7', 'PC8', 'PC9', 'PC10'], {'fontsize': 6})
    plt.colorbar(img)
    fig.savefig('{}/{}2_perturbation.svg'.format(folder, prefix, g))

def gradient_plot(gene, model_path, X, y, num_snps, class_weights, 
                        columns, num_covs, device, folder, prefix):
    """
    Function that perturbs each input in the data vector (one at a time) 
    for a neural network model and then calculates the change in loss 
    compared to the baseline model with no perturbations. It acts as a 
    proxy for the importance of elements in the data vector. A higher 
    increase in loss corresponds to greater importance.

    Parameters
    ----------
    gene : list of str
        List of gene names.
    model_path : list of str
        List of paths of saved PyTorch models for each gene.
    X : list of Numpy ndarray
        List of data for each gene.
    y : list of Numpy ndarray
        List of training labels for each gene.
    num_snps : list of int
        Number of SNPs in each gene.  
    class_weights : list of Numpy ndarray
        List of class weights for each gene for the loss function.
    columns : List of lists
        List of column headers for the data (SNPs, age, etc.) in each 
        gene.
    device : str
        Device for inference (eg: 'cuda:0').
    folder : str
        Base folder for saving generated plots.
    prefix : str
        Prefix to add to the file name while saving the plots.
    
    """
    if os.path.isfile('{}/{}_grads.svg'.format(folder, prefix)):
        return

    gs = []
    grads = np.ones((len(gene), max(num_snps)+num_covs))*-2
    sort_inds = np.ones((len(gene), max(num_snps)+num_covs))*np.nan
    # grads = np.ones((len(gene), 7))
    device = torch.device(device)
    min_val = 100
    max_val = -100
    
    for i, g in tqdm.tqdm(enumerate(gene)):
        gs.append(g)
        vprint('\t\t Gradients for {}'.format(g))
        
        snpsg = num_snps[i]
        colsg = columns[i]
        
        try:
            modelg = torch.load(model_path[i], map_location=device)
        except FileNotFoundError:
            print('Issue with ', model_path[i])
            continue
        cwg = torch.tensor(class_weights[i], device=device).float()
        loss_fng = nn.CrossEntropyLoss()
        Xg = torch.from_numpy(X[i]).float().to(device)
        Xg.requires_grad = True
        yg = torch.from_numpy(y[i]).long().to(device)
        
        modelg.eval()
        raw_out = modelg(Xg)
        loss = loss_fng(raw_out, yg)
        first_layer_weights = next(modelg.parameters())
        # for p in modelg.named_parameters():
        #     print(p[0], p[1].shape, p[1].requires_grad)
        
        grad = torch.autograd.grad(loss, Xg, 
            create_graph=True)
        
        for gr in grad:
            gr = torch.abs(gr)
            gr = torch.mean(torch.mean(gr, axis=0), axis=0)
            gr = (gr - torch.min(gr))/(torch.max(gr)-torch.min(gr))
            row = np.repeat(np.flip(gr.detach().cpu().numpy()), 1)
            min_val = min(min_val, np.min(row))
            max_val = max(max_val, np.max(row))
            sort_ind = np.arange(0, len(row))
            sort_ind[num_covs:] = np.flip(np.argsort(row[num_covs:])) - num_covs
            row[num_covs:] = np.flip(np.sort(row[num_covs:]))
            grads[i, :len(row)] = row
            sort_inds[i, :len(row)] = sort_ind
    
    grads = np.asarray(grads)[:, :num_covs+50].T
    cmap = copy.copy(plt.get_cmap('coolwarm'))
    cmap.set_under(alpha=0)
    plt.imshow(grads, cmap=cmap, origin='lower', vmin=0, vmax=1)
    plt.xticks(np.arange(len(gs)), gs, fontsize=2, rotation=90)
    # plt.yticks(np.arange(0, grads.shape[0], 1), columns[0][-12:-5], fontsize=4)
    plt.gca().axhline(num_covs-0.5, c='k', linestyle=':', linewidth=1)
    plt.title(prefix)
    plt.tight_layout()
    plt.savefig('{}/{}_grads.svg'.format(folder, prefix))
    plt.close()

    return grads, sort_inds

def gradient_pair_plot(gene, pval, model_path, X, y, num_snps, class_weights, 
                        columns, num_covs, device, folder, prefix):
    """
    Function that perturbs each input in the data vector (one at a time) 
    for a neural network model and then calculates the change in loss 
    compared to the baseline model with no perturbations. It acts as a 
    proxy for the importance of elements in the data vector. A higher 
    increase in loss corresponds to greater importance.

    Parameters
    ----------
    gene : list of str
        List of gene names.
    model_path : list of str
        List of paths of saved PyTorch models for each gene.
    X : list of Numpy ndarray
        List of data for each gene.
    y : list of Numpy ndarray
        List of training labels for each gene.
    num_snps : list of int
        Number of SNPs in each gene.  
    class_weights : list of Numpy ndarray
        List of class weights for each gene for the loss function.
    columns : List of lists
        List of column headers for the data (SNPs, age, etc.) in each 
        gene.
    device : str
        Device for inference (eg: 'cuda:0').
    folder : str
        Base folder for saving generated plots.
    prefix : str
        Prefix to add to the file name while saving the plots.
    
    """
    gs = []
    grads = np.ones((len(gene), max(num_snps)+num_covs))*-2
    # grads = np.ones((len(gene), 7))
    device = torch.device(device)
    for i, g in enumerate(gene):
        gs.append(g)
        print('\t\t Gradients for {}'.format(g))
        
        snpsg = num_snps[i]
        colsg = columns[i]
        gfold = '/'.join(model_path[i].split('/')[:-1])
        print(gfold)
        try:
            modelg = torch.load(model_path[i], map_location=device)
        except FileNotFoundError:
            print('Issue with ', model_path[i])
            continue
        cwg = torch.tensor(class_weights[i], device=device).float()
        loss_fng = nn.CrossEntropyLoss()
        Xg = torch.from_numpy(X[i]).float().to(device)
        Xg.requires_grad = True
        yg = torch.from_numpy(y[i]).long().to(device)
        
        modelg.eval()
        raw_out = modelg(Xg)
        loss = loss_fng(raw_out, yg)
        first_layer_weights = next(modelg.parameters())
        
        grad = torch.autograd.grad(loss, Xg, 
            create_graph=True, retain_graph=True)
        
        gr = grad[0]
        gr = torch.mean(torch.mean(gr, axis=0), axis=0)

        grad_mat = np.ones((len(gr), len(gr)))
        for gi in range(len(gr)):
            gr2 = torch.autograd.grad(gr[gi], Xg, retain_graph=True)[0]
            gr2 = torch.abs(gr2)
            gr2 = torch.mean(torch.mean(gr2, axis=0), axis=0)
            grad_mat[gi] = gr2.detach().cpu().numpy()
            grad_mat[gi, gi] = 0
            
        min_val = np.min(grad_mat)
        max_val = np.max(grad_mat)
        grad_mat = (grad_mat - min_val)/(max_val - min_val)
        grad_mat = np.triu(grad_mat)
        grad_mat = np.where(grad_mat!=0, grad_mat, np.ones(grad_mat.shape)*-1)
        grad_mat = np.where(grad_mat>=0.5, grad_mat, np.ones(grad_mat.shape)*-1)
            
        cmap = plt.get_cmap('bwr')
        cmap.set_under(alpha=0)
        plt.imshow(grad_mat, cmap=cmap, origin='lower', vmin=0, vmax=1)
        # plt.xticks(np.arange(len(gs)), gs, fontsize=2, rotation=90)
        # plt.yticks(np.arange(0, grads.shape[0], 1), columns[0][-12:-5], fontsize=4)
        plt.gca().yaxis.tick_right()
        plt.gca().axhline(len(grad_mat)-num_covs-0.5, c='lime', linestyle=':', 
            linewidth=2)
        plt.gca().axvline(len(grad_mat)-num_covs-0.5, c='lime', linestyle=':', 
            linewidth=2)
        plt.title('{} (p = {:.3e})'.format(g, pval[i]))
        plt.tight_layout()
        np.savez('{}/gradmats.npz'.format(gfold), grad_mat=grad_mat, 
            snps=colsg[:snpsg])
        plt.savefig('{}/gradmats.svg'.format(gfold))
        plt.close()

def epistasis_proxy(model_path, data, label, snps, gene, device, log, prefix):
    """
    Function that perturbs SNPs pairwise for a and then calculates the 
    change in loss compared to the baseline model with no perturbations. 
    It acts as a proxy for the importance of SNP interactions. A higher 
    increase in loss corresponds to greater importance. Finally 
    generates 2D heatplots representing interaction.

    Parameters
    ----------
    model_path: list of str
        List of paths of saved PyTorch models for each gene.
    data: list of Pandas DataFrame
        List of datasets for each gene.
    label: list of Numpy ndarray
        List of training labels for each gene.
    snps: list of tuples
        List of tuples of the form (num_snps, first_gene_snp, 
        last_gene_snp). The first_gene_snp and last_gene_snp are the 
        indices of the first and last snp on the gene. Useful to mark 
        the gene region on the plot, when flanking sequences on each 
        side of the gene are used (eg: 2500bp - APOE - 2500bp).
    gene: list of str
        List of gene names.
    device: str
        Device to train model on (eg: 'cuda:0').
    log: str
        Base folder for saving generated plots.
    prefix: str
        Prefix to add to the file name while saving the plots.

    """

    fname = '{}/{}_epistasis_{}.svg'.format(log, gene[0], prefix)
    if os.path.isfile(fname):
        return
    
    X = data[0]
    y = label[0]
    num_snps = snps[0][0]
    model = torch.load(model_path[0])
    model.to(torch.device(device))
    inp_labels = X.columns
    np.random.seed(8461)
    noise_vector = np.random.normal(0,1,X.shape[0])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.,1.]).cpu())

    l = np.zeros((num_snps, num_snps))
    a = np.zeros((num_snps, num_snps))

    # Without perturbation
    inp_data = torch.Tensor(X.copy().to_numpy()).to(torch.device(device))
    labels = torch.Tensor(y.copy()).cpu()
    model.eval()
    raw_out = model.forward(inp_data)
    y_pred = pred_from_raw(raw_out.detach().clone().cpu())
    loss = loss_fn(raw_out.cpu(), labels.long())
    _, _, _, acc, _ = metrics_from_raw(
                labels.detach().clone().cpu(), y_pred)
    l_mid = loss.item()
    a_mid = acc
    l_min = 100

    for i in tqdm.tqdm(range(0, num_snps)):
        for j in range(i, num_snps):
            # Add noise to every input column individually
            tmp = X.copy().to_numpy()
            if i == j:
                tmp[:, i] += noise_vector
            else:
                tmp[:, i] += noise_vector
                tmp[:, j] += noise_vector

            inp_data = torch.Tensor(tmp).to(torch.device(device))
            labels = torch.Tensor(y).cpu()
            
            model.eval()
            raw_out = model.forward(inp_data)
            y_pred = pred_from_raw(raw_out.detach().clone().cpu())
            loss = loss_fn(raw_out.cpu(), labels.long())

            f, p, r, acc, mcc = metrics_from_raw(
                labels.detach().clone().cpu(), y_pred)
        
            
            a[i][j] = acc
            l[i][j] = loss.item()
            l[j][i] = loss.item()
            l_min = min(l_min, l[j][i])

            del(tmp)
        
    fig, ax = plt.subplots()
    ax.axvline(snps[0][1]-0.5, c='k', linestyle=':')
    ax.axvline(snps[0][2]+0.5, c='k', linestyle=':')
    ax.axhline(snps[0][1]-0.5, c='k', linestyle=':')
    ax.axhline(snps[0][2]+0.5, c='k', linestyle=':')
    
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            if i == j:
                continue
            l_comb = l[i,j]
            # print('{:.6f} {:.6f}'.format(
            #   (0.68-l[i,j]), (1.36-l[i,i]-l[j,j])))
            l[i,j] = l_comb - max(l[i,i], l[j,j])
            # l[j,i] = l[i,j]
    
    for i in range(l.shape[0]):
        for j in range(i, l.shape[0]):
            if l[i,j] < 0:
                l[i,j] = l[j,i] = 0
        l[i,i] = 0
    
    cmap = cm.Reds
    cmap.set_under(alpha=0)
    # l = np.triu(l)
    vmin = np.min(l)
    print(vmin)
    vmax = 0.0082
    # l = np.tan(5*l)
    # vmin = np.tan(5*-0.01)
    # vmax = np.tan(5*0.01)
    # heatmap = ax.imshow(l, cmap=cmap, origin='lower')
    heatmap = ax.imshow(np.tril(l), cmap=cmap, origin='lower', vmin=1e-10)
    
    ticks = list(map(lambda x: x.split('_')[-1], inp_labels))
    if num_snps <= 20:
        lab_size = 8
    elif num_snps <= 40:
        lab_size = 6.5
    elif num_snps <= 60:
        lab_size = 5
    else:
        lab_size = 4

    ax.tick_params(axis='y', rotation=45, labelsize=lab_size)
    ax.set_yticks(np.arange(num_snps))
    ax.set_yticklabels(ticks[:num_snps])

    ax.tick_params(axis='x', rotation=90, labelsize=lab_size)
    ax.set_xticks(np.arange(num_snps))
    ax.set_xticklabels(ticks[:num_snps])
    # ax.xaxis.tick_top()

    fig.colorbar(mappable=heatmap)
    fig.tight_layout()
    fig.suptitle(gene[0]+': SNP-SNP Perturbation Loss Heatmap', fontsize=10)
    fig.subplots_adjust(top=0.9)
    fig.savefig(fname)
    plt.close(fig)


# Model construction and init functions
def construct_model(model_type, **kwargs):
    """ Helper function to create an object of a model class with the
    variable args needed for that model.

    Parameters
    ----------
    model_type : Model class
        The model class to initialise.

    Returns
    -------
    model
        Initialised model object.
    """
    return model_type(**kwargs)

def weight_init_linear(m):
    """
    Function to initialise model weights using Xavier uniform 
    initialisation based on a fixed random seed. It ensures that all 
    identical models are always initialised with the same weights.

    Parameters
    ----------
    m: torch Module object
        Model to initialise.

    """
    torch.manual_seed(0)
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(m.bias.data, -bound, bound)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(m.bias.data, -bound, bound)
    
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight.data)
        nn.init.zeros_(m.bias.data)


# General Training functions
def running_avg(old_avg:float, new_val:float, n:int) -> float:
    """Function to calculate the running average based on a new value,
    the old average and the size of the window.

    Parameters
    ----------
    old_avg : float
        The old average
    new_val : float
        The new value to be added to the old average
    n : int
        The window size to be used for the running average

    Returns
    -------
    float
        Running average over n samples.
    """
    return (old_avg*(n-1) + new_val)/n

def pred_from_raw(raw_out:torch.tensor) -> torch.tensor:
    """Function to convert raw outputs of the neural network into 
    predictions using argmax. Applies softmax to convert output to
    probabilitand then applies argmax to get the index of the output
    with the maximum probability.

    Parameters:
    ----------
    raw: torch tensor
        Output of neural network model.
    
    Returns
    -------
    torch tensor
        Class predictions.
    """
    pred = torch.argmax(torch.softmax(raw_out, dim=1), dim=1)
    return pred

def gen_conf_mat(y_true:torch.tensor, y_pred:torch.tensor, 
                 class_weights:torch.tensor) -> tuple:
    """Function to convert predictions of the neural network into 
    confusion matrix values. 

    Source: https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d#file-confusion_matrix_between_two_pytorch_tensors-py

    Parameters:
    ----------
    y_true: torch tensor
        Correct labels for each sample.
    y_pred: torch tensor
        Output of neural network model.
    class_weights : torch tensor (nclasses,)
        Weights to be used for each class.
    
    Returns
    -------
    tuple of float
        Confusion matrix entries (tn, fp, fn, tp).

    """

    # Element-wise subtraction after multiplication of y_pred by 2 returns a 
    # new tensor which holds a unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   2   where prediction is 1 and truth is 0 (False Positive)
    #   0   where prediction and truth are 0 (True Negative)
    #   -1     where prediction is 0 and truth is 1 (False Negative)
    confusion_vector = (y_pred*2) - y_true
    
    tn = torch.sum(confusion_vector == 0).item()*class_weights[1]
    fp = torch.sum(confusion_vector == 2).item()*class_weights[1]
    fn = torch.sum(confusion_vector == -1).item()*class_weights[0]
    tp = torch.sum(confusion_vector == 1).item()*class_weights[0]
    
    return (tn, fp, fn, tp)

def metrics_from_conf_mat(conf_mat:Union[tuple, list, np.ndarray]) -> tuple:
    """Function to convert confusion matrix values into F1-score, 
    precision, recall, accuracy and MCC. Any value is that not
    computable for a metric due to division by 0 or a 0/0 computation 
    is replaced by NaN. 

    Parameters
    ----------
    conf_mat: tuple, list or ndarray (4,)
        Confusion matrix entries (tn, fp, fn, tp).
    
    Returns
    -------
    tuple
        f1, prec, rec, acc, mcc
    """
    if isinstance(conf_mat, torch.Tensor):
        tn, fp, fn, tp = conf_mat.cpu()
    else:
        tn, fp, fn, tp = np.array(conf_mat)
    
    pos_pred = tp+fp
    neg_pred = tn+fn
    pos_obs = tp+fn
    neg_obs = tn+fp
    acc, prec, rec, mcc, f1 = np.nan, np.nan, np.nan, np.nan, np.nan
    if tp or tn or fp or fn:
        acc = (tp + tn)/(tn+fp+fn+tp)
    if pos_obs:
        rec = tp/pos_obs
    if pos_pred:
        prec = tp/pos_pred
    if not np.isnan(prec+rec) and (prec+rec):
        f1 = 2*((prec*rec)/(prec+rec))
    if pos_pred and neg_pred and pos_obs and neg_obs:
        mcc = ((tp*tn) - (fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return f1, prec, rec, acc, mcc

def train_val_loop(model:nn.Module, X:torch.tensor, y:torch.tensor, 
                   Xt:torch.tensor, yt:torch.tensor, training_dict:dict, 
                   log: str) -> tuple:
    """Model training and testing loop.

    Parameters:
    ----------
    model: torch.nn.Module
        Neural network model object.
    X: torch tensor
        Training data.
    y: torch tensor
        Training labels.
    Xt: torch tensor
        Testing data
    yt: torch tensor
        Testing labels
    training_dict: dict
        Dictionary containing training paramters with 
        the following keys:
        'model_name', 'train_ind', 'val_ind', 'loss_fn', 
        'optimiser', 'batch_size', 'epochs', 'scheduler',
        'device'
    log: str
        Folder to log training and best model in.
    
    Returns
    -------
    tuple
        Tuple of three:
            best_ep : Best epoch
            agg_conf_mat : torch tensor (epochs, 2, 4)
            avg_loss : torch tensor (epochs, 2)
    """
    # Get all training requirements from training_dict
    model_name = training_dict['model_name']
    train_ind = training_dict['train_ind']
    val_ind = training_dict['val_ind']
    loss_fn = training_dict['loss_fn']
    optimiser = training_dict['optimiser']
    batch_size = training_dict['batch_size']
    epochs = training_dict['epochs']
    scheduler = training_dict['scheduler']
    device = training_dict['device']
    class_weights = training_dict['class_weights']
    
    if Xt is not None:
        train_ind = np.concatenate((train_ind, val_ind))
        Xval, yval = Xt, yt
    else:
        Xval, yval = X[val_ind], y[val_ind]

    train_dataloader = FastTensorDataLoader(X[train_ind], y[train_ind],
        batch_size=batch_size, shuffle=True)
    train_inf_dataloader = FastTensorDataLoader(X[train_ind], y[train_ind],
        batch_size=8192, shuffle=False)
    val_dataloader = FastTensorDataLoader(Xval, yval, 
        batch_size=2048, shuffle=False)
    
    # Send model to device and initialise weights and metric tensors
    model.to(device)
    loss_fn = loss_fn.to(device)
    class_weights = torch.tensor(class_weights, device=device).float()
    best_ep = 0
    best_val = torch.tensor(0).float()
    agg_conf_mat = torch.zeros((epochs, 2, 4))
    avg_acc = torch.zeros((epochs, 2))
    avg_loss = torch.zeros((epochs, 2))
    
    current_lr = optimiser.state_dict()['param_groups'][0]['lr']
    best_state = model.state_dict()
    for epoch in tqdm.tqdm(range(epochs), desc='Epoch'):
        
        # Train
        model.train()
        for bnum, sample in enumerate(train_dataloader):
            model.zero_grad()
            
            X_batch = sample[0].to(device)
            y_batch = sample[1].long().to(device)

            raw_out = model.forward(X_batch)
            # y_pred = pred_from_raw(raw_out.detach().clone())
            loss = loss_fn(raw_out, y_batch)
            loss.backward()
            optimiser.step()

        # Infer
        for si, loader in enumerate([train_inf_dataloader, val_dataloader]):
            for bnum, sample in enumerate(loader):
                X_batch = sample[0].to(device)
                y_batch = sample[1].long().to(device)
                
                _, conf_mat, loss = infer(
                    X_batch, y_batch, model, loss_fn, device, 
                    class_weights=class_weights, batch_size=-1)
                agg_conf_mat[epoch][si] += torch.as_tensor(conf_mat[0])
                avg_loss[epoch][si] = running_avg(
                    avg_loss[epoch][si], loss[0], bnum+1)

            _, _, _, acc, _ = metrics_from_conf_mat(agg_conf_mat[epoch][si])
            avg_acc[epoch][si] = acc
        
        # If val acc plateaus or starts decreasing:
        # - Drop LR
        # - Backtrack to last best model and resume training
        if scheduler is not None:
            scheduler.step(avg_acc[epoch][1])
            new_lr = scheduler.optimizer.state_dict()['param_groups'][0]['lr']
            if new_lr < current_lr:
                model.load_state_dict(best_state)

        if log is not None:
            if best_val < avg_acc[epoch][1]:
                best_val = avg_acc[epoch][1]
                best_ep = epoch
                best_state = model.state_dict()
                torch.save(model, '{}/{}.pt'.format(log, model_name))
                # print("[{:4d}] Train Acc: {:.3f} Val Acc: {:.3f}, \
                #             Train Loss: {:.3f} Val Loss:{:.3f}".format(
                #                 epoch, avg_acc[epoch][0], avg_acc[epoch][1], 
                #                 avg_loss[epoch][0], avg_loss[epoch][1]))
            
            if epoch%200 == 0 or (epoch == epochs-1):
                torch.save(model, '{}/{}_Ep{}.pt'.format(log, model_name, epoch))
    
    print('\n\n', model_name, ' BEST EPOCH ', best_ep, '\n\n')
    
    return best_ep, agg_conf_mat, avg_loss

def training_stuff(model:nn.Module, damping:float, class_weights:np.ndarray, 
                   lr:float, opt:str) -> tuple:
    """Function to return objects needed to train the neural network.

    Parameters:
    ----------
    model: torch.nn.Module
        Neural network model object.
    damping: float
        Damping value for custom loss function (Still being worked on. 
        Best to use the value 1.0 for now, which corresponds 
        to Cross-Entropy Loss).
    class_weights: list of float
        Class weights for training with imbalanced classes.
    lr: float
        Learning rate.
    opt: str
        Optimiser to use while training the neural network.

    Returns
    -------
    tuple
        Three element tuple:
            Loss class object to use while training
            Optimiser class object to use while training
            Learning rate scheduler object to use while training
    """
    # LOSS FUNTION
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
    # OPTIMISER
    if opt == 'sgd':
        optimiser = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
            momentum=0.9, lr=lr)
    if opt == 'rmsprop':
        optimiser = optim.RMSprop(model.parameters(), lr=lr)
    if opt == 'adam':
        optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr)
    if opt == 'adamw':
        optimiser = optim.AdamW(model.parameters(), lr=lr)
    if opt == 'adagrad':
        optimiser = optim.Adagrad(model.parameters(), lr=lr)
    
    # SCHEDULER
    scheduler = ReduceLROnPlateau(optimiser, mode='max', patience=50, 
        factor=0.5, min_lr=1e-6, verbose=True)
    
    return loss_fn, optimiser, scheduler

def start_training(X:np.ndarray, y:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, 
                   model_dict:dict, optim_dict:dict, train_dict:dict, 
                   device:Union[str, int]) -> tuple:
    """Helper function used to start model training.

    Parameters
    ----------
    X : np.ndarray
        Training data.
    y : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    model_dict : dict
        Dictionary with model parameters.
    optim_dict : dict
        Dictionary with optimiser parameters.
    train_dict : dict
        Dictionary with training parameters.
    device : Union[int, str]
        GPU to train on. Should be a valid argument for torch.device().

    Returns
    -------
    tuple
        Two element tuple:
            conf_mat : numpy ndarray (nepochs, 2, 4)
            loss : numpy ndarray (nepochs, 2)
    """

    # Load model parameters
    model_name = model_dict['model_name']
    model_type = model_dict['model_type']
    model_args = model_dict['model_args']

    # Load model optimiser paramters
    lr = optim_dict['LR']
    damp = optim_dict['damping']
    class_weights = optim_dict['class_weights']
    class_weights = torch.Tensor(class_weights).cpu()
    optimiser = optim_dict['optim']
    use_scheduler = optim_dict['use_scheduler']
    
    # Load training parameters
    batch_size = train_dict['batch_size']
    epochs = train_dict['epochs']
    log = train_dict['log']

    # Convert numpy data into torch tensors
    X, y = torch.tensor(X).float(), torch.tensor(y).long()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).long()
    
    if 'pretrained_model' in model_dict.keys():
        model = torch.load(model_dict['pretrained_model'], 
            map_location='cpu')
    else:
        model = construct_model(model_type, **model_args)
        model.apply(weight_init_linear)
    
    loss_fn, optimiser, scheduler = training_stuff(model=model, damping=damp, 
                                        class_weights=class_weights, lr=lr, 
                                        opt=optimiser)
    if not use_scheduler:
        scheduler = None

    train_ind = np.arange(X.shape[0])
    np.random.seed(6211)
    np.random.shuffle(train_ind)

    training_dict = {
        'model_name': model_name,
        'train_ind': train_ind,
        'val_ind': np.array([], dtype=int),
        'loss_fn': loss_fn,
        'optimiser':optimiser,
        'scheduler': scheduler,
        'batch_size': batch_size,
        'epochs':epochs,
        'class_weights':class_weights,
        'device': device
    }
    
    return train_val_loop(model=model, X=X, y=y, Xt=X_test, yt=y_test, 
                        training_dict=training_dict, log=log)

def infer(X_tensor:torch.tensor, y_tensor:torch.tensor, model:nn.Module, 
          loss_fn:nn.Module, device:Union[str, int], perms:Optional[list]=None, 
          num_snps:int=0, class_weights:list=[1,1], batch_size:int=256) -> tuple:
    """Model inference.

    Parameters:
    ----------
    X_tensor : torch.tensor
        Training data.
    y_tensor : torch.tensor
        Training labels.
    model : nn.Module
        NN model.
    loss_fn : PyTorch loss object
        Loss function object. 
    device : str
        GPU/CPU name.
    perms : list, optional
        List of permuted indices. This should be a 2D list or None.
    num_snps : int, optional
        Number of SNPs in the data.
    class_weigths: list (nclasses,), optional
        Class weights used in calculating the metrics (default=[1,1]
        representing balanced binary classification)
    batch_size : int, optional
        Batch size to use for the dataloader. If -1, will set the batch
        size to the size of the data. 

    Returns
    -------
    tuple
        y_pred : Numpy ndarray
            Network predictions.
        conf_mat : tuple of float
            (tn, fp, fn, tp)
        loss : float
            Loss.
    """
    if perms is None:
        perms = [torch.arange(0, X_tensor.shape[0]),]

    if batch_size == -1:
        batch_size = X_tensor.shape[0]
    
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    conf_mats = []
    y_preds = []
    losses = []
    
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for perm in perms:
            X_ = X_tensor.clone()
            X_[:, :, :num_snps] = X_[perm, :, :num_snps]

            dataloader = FastTensorDataLoader(X_, y_tensor, batch_size=batch_size)
            y_pred = torch.tensor([], device=device).long()
            loss = 0.0
            
            for bnum, sample in enumerate(dataloader):
                X_batch = sample[0]
                y_batch = sample[1]
                
                model.eval()
                model = model.to(device)
                loss_fn = loss_fn.to(device)
                
                raw_out = model.forward(X_batch)
                y_pred = torch.cat(
                    (y_pred, pred_from_raw(raw_out.detach().clone())))
                batch_loss = loss_fn(raw_out, y_batch).detach().item()
                loss = running_avg(loss, batch_loss, bnum+1)
    
            if len(y_tensor) != len(y_pred):
                diff = len(y_tensor) - len(y_pred)
                conf_mat = gen_conf_mat(y_tensor[:-diff].detach().clone(), y_pred,
                    class_weights=class_weights)
            else:
                conf_mat = gen_conf_mat(y_tensor.detach().clone(), 
                    y_pred.to(y_tensor.device), class_weights=class_weights)
            
            y_preds.append(y_pred.cpu())
            losses.append(loss)
            conf_mats.append(list(conf_mat))
            del X_
    torch.backends.cudnn.benchmark = False

    return y_preds, conf_mats, losses

def train(X:np.ndarray, y:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, 
          model_dict:dict, optim_dict:dict, train_dict:dict, 
          device:Union[int, str]) -> tuple:
    """Invoke training and save aggregated confusion matrix and loss.

    Parameters
    ----------
    X : np.ndarray
        Training data.
    y : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    model_dict : dict
        Dictionary with model parameters.
    optim_dict : dict
        Dictionary with optimiser parameters.
    train_dict : dict
        Dictionary with training parameters.
    device : Union[int, str]
        GPU to train on. Should be a valid argument for torch.device().
    
    Returns
    ----------
    tuple
        Tuple of three:
            Best epoch
            Best test set accuracy 
            Best train set accuracy
    """
    
    best_ep, conf_mat, loss = start_training(
        X=X, y=y, X_test=X_test, y_test=y_test, model_dict=model_dict, 
        optim_dict=optim_dict, train_dict=train_dict, device=device)
    if train_dict['log'] is not None:
        metric_logs = f'{train_dict["log"]}/training_metrics.npz'
        np.savez(metric_logs, agg_conf_mat=conf_mat, agg_loss=loss)
    
    best_test_cm = conf_mat[best_ep, 1]
    metrics = metrics_from_conf_mat(best_test_cm)
    best_test_acc = metrics[3].item()
    best_test_loss = loss[best_ep, 1].item()

    return best_ep, best_test_acc, best_test_loss


# Permutatation Testing functions
def case_control_permutations(labels, k, start=0):
    """
    Return k permutations of the indices (each one different from the 
    other). The indices will only be shuffled between case and control. 
    Hence, after permuting case indices, they will be assigned to 
    control and after vice versa. This ensures that no case (control) 
    individual is assigned the data of another case (control) 
    individual. 

    Parameters
    ----------
    labels: list
        List of data labels.
    k: int
        Number of permutations.
    start : int 
        Used to initialise the random number seed. Also if start=0, the
        permutation is just the range(len(labels)), so it is equivalent
        to the original order without any shuffling. 

    Returns
    -------
    numpy ndarray
        List of k permutations of the indices.
    """

    n = len(labels)
    case_ind = np.where(labels == 1)[0]
    control_ind = np.where(labels == 0)[0]

    perms = np.empty((k, n), dtype=int)
    
    np.random.seed(1042343)
    random_seeds = np.random.choice(np.arange(0, int(1e6)), start+k, replace=False)
    random_seeds = random_seeds[start:]

    for i in range(0, k):
        np.random.seed(random_seeds[i])
        case_perm = np.random.permutation(case_ind)
        np.random.seed(random_seeds[i])
        control_perm = np.random.permutation(control_ind)        
        
        # Make sure no case is in control and vice versa
        assert len(np.intersect1d(case_perm, control_ind)) == 0
        assert len(np.intersect1d(control_perm, case_ind)) == 0
        
        perms[i, case_ind] = case_perm
        if len(case_ind) < len(control_perm):
            perms[i, case_ind] = control_perm[:len(case_ind)]
        else:
            perms[i, case_ind[:len(control_perm)]] = control_perm
        
        perms[i, control_ind] = control_perm
        if len(control_ind) < len(case_perm):
            perms[i, control_ind] = case_perm[:len(control_ind)]
        else:
            perms[i, control_ind[:len(case_perm)]] = case_perm
        
        # Make sure that no sample appears twice in a permutation
        assert len(np.unique(perms[i])) == len(perms[i])

    if start == 0:
        perms[0] = np.arange(n, dtype=int)

    # Make sure that no 2 permutations are the same 
    num_unique_perms = len(np.unique(perms, axis=0))
    assert num_unique_perms == k

    return perms

def random_permutations(labels, k, start=0):
    """
    Return k random permutations of the indices (each one different from 
    the other).

    Parameters
    ----------
    labels: list
        List of data labels.
    k: int
        Number of permutations.
    start : int 
        Used to initialise the random number seed. Also if start=0, the
        permutation is just the range(len(labels)), so it is equivalent
        to the original order without any shuffling. 

    Returns
    -------
    numpy ndarray
        List of k permutations of the indices.
    """
    n = len(labels)
    perms = np.empty((k, n), dtype=int)

    for i in range(0, k):
        np.random.seed(start+i)
        perms[i] = np.random.permutation(n)
    if start == 0:
        perms[0] = np.arange(n, dtype=int)
    # Make sure that no 2 permutations are the same 
    num_unique_perms = len(np.unique(perms, axis=0))
    assert(num_unique_perms == k)

    return perms

def modified_permutation_test(model_path, X, y, X_test, y_test, num_snps,
                            ptest_dict, device, class_weights=[1,1], 
                            batch_size=-1):
    """Function to run the modified version of the permutation test.
    Instead of training the model on permuted data every single time,
    the model is trained on the unpermuted data and then tested on
    various permuted combinations. 

    Parameters
    ----------
    model_path : str
        Path to saved model to be used for inference.
    X : ndarray
        Training data to be permuted.
    y : ndarray
        Training labels.
    X_test : ndarray
        Testing data.
    y_test : ndarray
        Testing labels.
    num_snps : int
        Number of SNPs in the data
    ptest_dict : dict
        Dictionary containing the {num_perm, perm_method}
    devices : list of str
        List of GPUs to be used for the test.
    class_weights : list or ndarray
        Weights for each class to be used while calculating the metrics
        and creating the loss function (default [1,1])

    Returns
    -------
    p_values : dict
        Dictionary of p-values calculated based on the different metrics
    conf_mat : ndarray (nperms, 4)
        Confusion matrix for unpermuted (index 0) and permuted data
    f1 : ndarray (nperms,)
        F1 scores for unpermuted (index 0) and permuted data
    prec : ndarray (nperms,)
        Precision for unpermuted (index 0) and permuted data.
    rec : ndarray (nperms,)
        Recall for unpermuted (index 0) and permuted data.
    acc : ndarray (nperms,)
        Accuracy for unpermuted (index 0) and permuted data.
    mcc : ndarray (nperms,)
        MCC for unpermuted (index 0) and permuted data.
    loss : ndarray (nperms,)
        Loss for unpermuted (index 0) and permuted data.
    """
    # Load permutation test parameters
    num_perm = ptest_dict['num_perm']
    perm_method = ptest_dict['perm_method']
    
    if X_test is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, 
            random_state=447)
        for train_ind, val_ind in splitter.split(X, y):
            X = X[val_ind]
            y = y[val_ind]
    else:
        X = X_test
        y = y_test
    
    # Zeroing out covariates for inference
    # X[:, :, num_snps:] = np.random.normal(0, 1, (X[:, :, num_snps:].shape))
    # print(X[0])
    
    class_weights = torch.Tensor(class_weights)

    # Test model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    func_args = []
    agg_conf_mat = []
    agg_loss = []

    # Run the inference for all permutations
    s1 = datetime.datetime.now()
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    chunk_size = 1000
    print('{} Starting'.format(model_path.split('/')[-1]))
    for chunk in range(num_perm//chunk_size):
        # permutations = perm_method(np.repeat(y, X.shape[1]), chunk_size, chunk_size*(chunk))
        permutations = perm_method(y, chunk_size, start=chunk_size*(chunk))
        _, cm, l = infer(X, y, model, loss_fn, device, 
            permutations, num_snps, class_weights, batch_size)
        agg_conf_mat.extend(cm)
        agg_loss.extend(l)
        print('{} Chunk {}'.format(model_path.split('/')[-1], chunk))
    e1 = datetime.datetime.now()
    print('\tInference time: ', (e1-s1))
    
    # Calculate all metrics from the confusion matrix
    f1 = np.empty((num_perm,))
    prec = np.empty((num_perm,))
    rec = np.empty((num_perm,))
    acc = np.empty((num_perm,))
    mcc = np.empty((num_perm,))
    loss = np.array(agg_loss)    
    for i, cm in enumerate(agg_conf_mat):
        f1[i], prec[i], rec[i], acc[i], mcc[i] = metrics_from_conf_mat(cm)

    print(model_path.split('/')[-1], acc[0])
    
    # Get p-values based on different metrics
    p_f1 = np.count_nonzero(f1 >= f1[0])/num_perm
    p_prec = np.count_nonzero(prec >= prec[0])/num_perm
    p_rec = np.count_nonzero(rec >= rec[0])/num_perm
    p_acc = np.count_nonzero(acc >= acc[0])/num_perm 
    p_mcc = np.count_nonzero(mcc >= mcc[0])/num_perm
    p_loss = np.count_nonzero(loss <= loss[0])/num_perm
    p_values = {'p_f1':p_f1, 'p_prec':p_prec, 'p_rec':p_rec, 'p_acc':p_acc,
        'p_mcc':p_mcc, 'p_loss':p_loss}

    return p_values, np.array(agg_conf_mat), f1, prec, rec, acc, mcc, loss
    
def hundredMillion_ptest(model_path, X, y, X_test, y_test, num_snps,
                            ptest_dict, class_weights=[1,1], 
                            batch_size=-1, device=0, perm_range=(0,0)):
    
    # Load permutation test parameters
    num_perm = perm_range[1] - perm_range[0]
    print(num_perm)
    perm_method = ptest_dict['perm_method']
    
    if X_test is None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, 
            random_state=447)
        for train_ind, val_ind in splitter.split(X, y):
            X = X[val_ind]
            y = y[val_ind]
    else:
        X = X_test
        y = y_test
    
    class_weights = torch.Tensor(class_weights)

    # Test model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    func_args = []
    agg_conf_mat = []
    agg_loss = []

    # Run the inference for all permutations
    s1 = datetime.datetime.now()
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    chunk_size = int(1e2)
    print('{} Starting'.format(model_path.split('/')[-1]))
    for chunk in range(num_perm//chunk_size):
        # permutations = perm_method(np.repeat(y, X.shape[1]), chunk_size, chunk_size*(chunk))
        permutations = perm_method(y, chunk_size, 
            start=perm_range[0] + chunk_size*(chunk))
        _, cm, l = infer(X, y, model, loss_fn, device, 
            permutations, num_snps, class_weights, batch_size)
        agg_conf_mat.extend(cm)
        agg_loss.extend(l)
        print('{} Chunk {}'.format(model_path.split('/')[-1], chunk))
    e1 = datetime.datetime.now()
    print('\tInference time: ', (e1-s1))
    
    # Calculate all metrics from the confusion matrix
    f1 = np.empty((num_perm,))
    prec = np.empty((num_perm,))
    rec = np.empty((num_perm,))
    acc = np.empty((num_perm,))
    mcc = np.empty((num_perm,))
    loss = np.array(agg_loss)    
    for i, cm in enumerate(agg_conf_mat):
        f1[i], prec[i], rec[i], acc[i], mcc[i] = metrics_from_conf_mat(cm)
    
    print(perm_range[0] + chunk_size*(chunk), acc[0])
    
    p_values = None

    return p_values, np.array(agg_conf_mat), f1, prec, rec, acc, mcc, loss
    # return (None, np.ones((int(1e7), 4)), np.ones(int(1e7)), 
    #     np.ones(int(1e7)), np.ones(int(1e7)), np.ones(int(1e7)), 
    #     np.ones(int(1e7)), np.ones(int(1e7)))

# Hyperparamter tuning/search functions
def hyperparam_search(X, y, num_comb, class_weights, hyperparams, log, devices, 
                        grid):
    """
    Function to run hyperparameter search for a gener and 
    log metrics for each model.

    Parameters:
    ----------
    X: numpy ndarray
        Training data.
    y: numpy ndarray
        Training labels.
    num_comb: int
        Number of hyperparameter combinations to try.
    hyperparams: dict
        Dictionary containing different hyperparameter values.
    log: str
        File to log model metrics in.
    devices: list of str
        List of availaible CPUs/GPUs to run the different 
        models on.
    grid: bool
        Grid search if True, random search if False.
            
    """

    hidden = hyperparams['hidden']
    dropout = hyperparams['dropout']
    activ = hyperparams['activ']
    lr = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    optim = hyperparams['optim']
    epochs = hyperparams['epochs']
    f1_method = hyperparams['f1_method']
    
    done_combs = []
    # Write csv file headers if the log file does not exist 
    if not os.path.isfile(log):
        with open(log, mode='w') as logfile:
            wr = csv.writer(logfile)
            wr.writerow(['hidden', 'dropout', 'activ', 'LR', 
                        'Optim', 'Batch', 'Epochs', 'F1_meth',
                        'train_f1', 'train_acc', 'train_loss',
                        'val_f1', 'val_acc', 'val_loss'])
    else:
        with open(log, mode='r') as logfile:
            r = csv.reader(logfile)
            done_combs = list(r)[1:]
        done_combs = [ast.literal_eval(x[0]) for x in done_combs]

    splitter = StratifiedShuffleSplit(n_splits=1, 
                                    test_size=0.1,
                                    random_state=100)
    num_devices = len(devices)
    cnt = 0
    func_args = []
    hyps = []

    # Implement random selection of hyperparameter combination
    comb_all = list(it.product(hidden, dropout, activ, 
                                lr, optim, batch_size, 
                                epochs, f1_method))
    if grid:
        comb = comb_all
    else:
        np.random.seed(9721)
        comb_ind = np.random.choice(
            np.arange(0, len(comb_all)), num_comb, replace=False)
        comb = [comb_all[i] for i in comb_ind]
    

    print('Total combinations: ', len(comb))
    for ind, hyp in enumerate(comb):
        if hyp[0] in done_combs:
            print('Skipping {} because already done'.format(hyp[0]))
            continue
        
        # Get copy of the data
        X_ = X.copy()
        y_ = y.copy()

        # Model Parameters
        model_dict = {}
        model_dict['model_name'] = ''
        model_dict['model_type'] = BasicNN
        model_dict['model_args'] = {'inp':X_.shape[1],
                                    'h':hyp[0],
                                    'd':list(np.repeat(hyp[1], len(hyp[0]))),
                                    'out':2,
                                    'activation':hyp[2]
                                    }
        # Optimiser Parameters
        optim_dict = {}
        optim_dict['LR'] = hyp[3]
        optim_dict['damping'] = 1.0
        optim_dict['class_weights'] = class_weights
        optim_dict['optim'] = hyp[4]
        optim_dict['use_scheduler'] = False
        # Training Parameters
        train_dict = {}
        train_dict['batch_size'] = hyp[5]
        train_dict['epochs'] = hyp[6]
        train_dict['f1_avg'] = hyp[7]
        train_dict['log'] = None
        
        # Accumulate function arguments
        if cnt < num_devices:
            device = 'cuda:' + str(devices[ind%num_devices])
            print('Device: {}  Combination: {}'.format(device, ind))
            func_args.append((X_, y_, 
                            None, None, 
                            splitter, 
                            train_val_loop, 
                            model_dict, 
                            optim_dict,
                            train_dict,
                            device,
                            1))
            cnt+=1
            hyps.append(list(hyp))
        
        # Run function in parallel
        if cnt == num_devices or ind+1 == len(comb):
            with mp.Pool(num_devices) as pool:
                results = pool.starmap(start_training, func_args, chunksize=1)
                for rnum, r in enumerate(results):
                    for key in ['train', 'val']:
                        hyps[rnum].extend([r[0][key], r[1][key], r[2][key]])

                    with open(log, mode='a') as logfile:
                        wr = csv.writer(logfile)
                        wr.writerow(hyps[rnum])
                    
                func_args = []
                hyps = []
                cnt = 0
    
def pick_best_model():
    """Return a model dictionary with the best model architecture.

    """

    model_dict = {
        'hidden': [128,64,16],
        'drop': 0.3,
        'optimiser': 'adam',
        'activ': nn.ReLU,
        'lr': 1e-3,
        'batch': 128,
        'epochs': 200,
    }

    return model_dict
