
# coding: utf-8
import os
import csv
import ast
from scipy.stats.stats import mode
from sympy import rotations
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

shared_data = {}

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

class ModifiedCELoss_(nn.Module):
    """
    The loss is a modified version of the cross entropy loss. The input 
    to the loss is soft=min(softmax(y), damping) where damping is the
    prediction probability we are satisfied with. In cross entropy, the
    damping is 1.0 by default. The loss is finally calculated as
    -ln(soft/damping) so that the loss for a predcition probability of
    >=damping is considered as 0.

    """
    def __init__(self, damping, class_weights=None):
        super(ModifiedCELoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.damping = damping
        self.weight = class_weights

    def forward(self, y_raw, y_true):        
        # Apply softmax (~sigmoid if 2 classes) and find network prediciton
        dev = y_raw.device
        soft = self.softmax(y_raw)
        soft = torch.gather(soft, 1, y_true.view(-1,1)).flatten()
        damping_tensor = torch.tensor(
            self.damping, device=dev).repeat(len(soft),1).flatten()
        soft = torch.min(soft, damping_tensor)
        if self.weight is not None:
            weight_vector = torch.Tensor(self.weight[y_true.long()])
        else:
            weight_vector = torch.ones(len(y_true))
        nll = -weight_vector*torch.log(soft/self.damping)
        nll = torch.mean(nll)
        return nll

class KDLoss_(nn.Module):
    """
    Knowledge Distillation Loss (Hinton et. al.)
    ref: https://arxiv.org/abs/1503.02531

    """
    def __init__(self, T, alpha):
        super(KDLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = alpha
        self.T = T
        self.ce = nn.CrossEntropyLoss()

    def forward(self, zS, zT, yT):
        hard_loss = self.ce(zS, yT)
        soft_loss = self.T**2 * self.ce(zS/self.T, zT/self.T)
        return (1-self.alpha)*hard_loss + self.alpha*soft_loss

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
        ax.plot(x, yv, label=m+'_train', linestyle=':', c=colors[m])
        ax.tick_params(axis='both', labelsize=8)
        
    ax.set_xlabel('Epochs')
    ax.set_ylabel('_'.join(m_plot))
    ax.legend()
    fig.savefig(fig_name)
    plt.close(fig)
              
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
    # print(m)
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

def running_avg(old_avg, new_val, n):
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

def pred_from_raw(raw):
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
    pred : torch tensor
        Class predictions.

    """
    pred = torch.argmax(torch.softmax(raw, dim=1), dim=1)
    return pred

def gen_conf_mat(y_true, y_pred, class_weights):
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

def metrics_from_conf_mat(conf_mat):
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
    f1, prec, rec, acc, mcc : float
        F1, precision, recall, accuracy and MCC metrics.

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

def training_stuff(model, damping, class_weights, lr, opt):
    """Function to return objects needed to train the neural network.

    Parameters:
    ----------
    model: torch Module
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
    Loss class object to use while training
    Optimiser class object to use while training
    Learning rate scheduler object to use while training
        
    """
    # LOSS FUNTION
    if damping == 1:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = ModifiedCELoss(damping, class_weights)
        
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

def train_val_loop(model, X, y, Xt, yt, training_dict, log):
    """Model training and validation loop.

    Parameters:
    ----------
    model: torch Module
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
    agg_conf_mat : torch tensor (epochs, 2, 4)
        Validation set confusion matrix values (tn, fp, fn, tp) for 
        every epoch.
    """
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # Get all training requirements from training_dict
    model_name = training_dict['model_name']
    train_ind = training_dict['train_ind']
    val_ind = training_dict['val_ind']
    loss_fn = training_dict['loss_fn']
    optimiser = training_dict['optimiser']
    batch_size = training_dict['batch_size']
    epochs = training_dict['epochs']
    scheduler = training_dict['scheduler']
    devices = training_dict['device']
    device = devices[0]
    class_weights = training_dict['class_weights']
    # early_stopping = EarlyStopping(
    #     patience=20, path='{}/{}.pt'.format(log, model_name))

    if Xt is not None:
        train_ind = np.concatenate((train_ind, val_ind))
        Xval, yval = Xt, yt
    else:
        Xval, yval = X[val_ind], y[val_ind]

    train_dataloader = FastTensorDataLoader(X[train_ind], y[train_ind],
        batch_size=batch_size, shuffle=False)
    train_inf_dataloader = FastTensorDataLoader(X[train_ind], y[train_ind],
        batch_size=8192, shuffle=False)
    val_dataloader = FastTensorDataLoader(Xval, yval, 
        batch_size=2048, shuffle=False)
    
    # Send model to device and initialise weights and metric tensors
    # model = nn.DataParallel(model, device_ids=devices)
    print('DEVICE:', device)
    model.to(device)
    model.apply(weight_init_linear)
    loss_fn = loss_fn.to(device)
    class_weights = torch.tensor(class_weights, device=device).float()
    best_ep = 0
    best_val = torch.tensor(0).float()
    agg_conf_mat = torch.zeros((epochs, 2, 4))
    avg_f1 = torch.zeros((epochs, 2))
    avg_prec = torch.zeros((epochs, 2))
    avg_rec = torch.zeros((epochs, 2))
    avg_acc = torch.zeros((epochs, 2))
    avg_mcc = torch.zeros((epochs, 2))
    avg_loss = torch.zeros((epochs, 2))
    
    current_lr = optimiser.state_dict()['param_groups'][0]['lr']
    best_state = model.state_dict()
    for epoch in range(epochs):
        # Train loop
        for bnum, sample in enumerate(train_dataloader):
            model.train()
            model.zero_grad()
            
            X_batch = sample[0].to(device)
            y_batch = sample[1].long().to(device)

            raw_out = model.forward(X_batch)
            y_pred = pred_from_raw(raw_out.detach().clone())
            loss = loss_fn(raw_out, y_batch)
            loss.backward()
            optimiser.step()

        # Train inference
        for bnum, sample in enumerate(train_inf_dataloader):
            X_batch = sample[0].to(device)
            y_batch = sample[1].long().to(device)
            
            _, conf_mat, loss = infer(
                X_batch, y_batch, model, loss_fn, device, 
                class_weights=class_weights, batch_size=-1)
            agg_conf_mat[epoch][0] += torch.as_tensor(conf_mat[0])
            avg_loss[epoch][0] = running_avg(
                avg_loss[epoch][0], loss[0], bnum+1)

        f, p, r, acc, mcc = metrics_from_conf_mat(agg_conf_mat[epoch][0])
        avg_f1[epoch][0] = f
        avg_prec[epoch][0] = p
        avg_rec[epoch][0] = r
        avg_acc[epoch][0] = acc
        avg_mcc[epoch][0] = mcc
        
        # Val loop
        for bnum, sample in enumerate(val_dataloader):
            X_batch = sample[0].to(device)
            y_batch = sample[1].long().to(device)
            
            _, conf_mat, loss = infer(
                X_batch, y_batch, model, loss_fn, device, 
                class_weights=class_weights, batch_size=-1)
            agg_conf_mat[epoch][1] += torch.as_tensor(conf_mat[0])
            avg_loss[epoch][1] = running_avg(
                avg_loss[epoch][1], loss[0], bnum+1)

        f, p, r, acc, mcc = metrics_from_conf_mat(agg_conf_mat[epoch][1])
        avg_f1[epoch][1] = f
        avg_prec[epoch][1] = p
        avg_rec[epoch][1] = r
        avg_acc[epoch][1] = acc
        avg_mcc[epoch][1] = mcc

        # If val acc plateaus or starts decreasing:
        # - Drop LR
        # - Backtrack to last best model and resume training
        if scheduler is not None:
            scheduler.step(avg_acc[epoch][1])
            new_lr = scheduler.optimizer.state_dict()['param_groups'][0]['lr']
            if new_lr < current_lr:
                model.load_state_dict(best_state)

        print("[{:4d}] Train Acc: {:.3f} Val Acc: {:.3f}, \
                Train F1: {:.3f} Val F1: {:.3f} \
                Train MCC: {:.3f} Val MCC:{:.3f} \
                Train Loss: {:.3f} Val Loss:{:.3f}".
              format(epoch, avg_acc[epoch][0], avg_acc[epoch][1], 
                     avg_f1[epoch][0], avg_f1[epoch][1],
                     avg_mcc[epoch][0], avg_mcc[epoch][1],
                     avg_loss[epoch][0], avg_loss[epoch][1]))

        if log is not None:
            if best_val < avg_acc[epoch][1]:
                best_val = avg_acc[epoch][1]
                best_ep = epoch
                best_state = model.state_dict()
                torch.save(model, '{}/{}.pt'.format(log, model_name))
            if epoch%200 == 0 or (epoch == epochs-1):
                torch.save(model, '{}/{}_Ep{}.pt'.format(log, model_name, epoch))
    
    print('\n\n', model_name, ' BEST EPOCH ', best_ep, '\n\n')
    
    return agg_conf_mat, avg_loss

def start_training(X, y, X_test, y_test, splitter, train_func, 
                    model_dict, optim_dict, train_dict, 
                    device, perm):
    """Helper function used to start model training.

    Parameters:
    ----------
    X: numpy ndarray
        Training data.
    y: numpy ndarray
        Training labels.
    X_test: numpy ndarray
        Testing data
    y_test: numpy ndarray
        Testing labels
    splitter: sklearn splitter object
        Object to use while splitting data into train 
        and validation set. Decides whether to use 1 fold 
        or 10 fold cv.
    train_func: function 
        Function to use to train the network
    model_dict: dict
        Dictionary containing paramters to create 
        neural network model.
    optim_dict: dict
        Dictionary containing paramters to create 
        training optimiser.
    train_dict: dict
        Dictionary containing training paramters
    device: str
        GPU or CPU to run the model on.
    perm: int
        Permutation number in case of permutation test.
    
    Returns
    -------
    agg_conf_mat : numpy ndarray (nfolds, nepochs, 2, 4)
    agg_loss : numpy ndarray (nfolds, nepochs, 2)
            
    """

    # Load parameters
    model_name = model_dict['model_name']
    model_type = model_dict['model_type']
    model_args = model_dict['model_args']

    # Load model optimiser paramters
    lr = optim_dict['LR']
    damp = optim_dict['damping']
    class_weights = optim_dict['class_weights']
    class_weights = torch.Tensor(class_weights).cpu()
    optim = optim_dict['optim']
    use_scheduler = optim_dict['use_scheduler']
    
    # Load training parameters
    batch_size = train_dict['batch_size']
    epochs = train_dict['epochs']
    log = train_dict['log']

    # Convert numpy data into torch tensors
    X, y = torch.tensor(X).float(), torch.tensor(y).long()
    if X_test is not None:
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).long()

    fold_num = 0
    log_ = log
    agg_conf_mat = []
    agg_loss = []
    for train_ind, val_ind in splitter.split(X, y):
        print('FOLD ', fold_num)
        print('**************************')
        
        if 'pretrained_model' in model_dict.keys():
            model = torch.load(model_dict['pretrained_model'], 
                map_location='cpu')
        else:
            model = construct_model(model_type, **model_args)
        
        loss_fn, optimiser, scheduler = training_stuff(model, damp,
                                                       class_weights,
                                                       lr, optim)
        if not use_scheduler:
            scheduler = None

        training_dict = {
            'model_name': model_name,
            'train_ind': train_ind,
            'val_ind': val_ind,
            'loss_fn': loss_fn,
            'optimiser':optimiser,
            'scheduler': scheduler,
            'batch_size': batch_size,
            'epochs':epochs,
            'class_weights':class_weights,
            'device': device
        }
        if fold_num != 0 or perm != 0:
            log_ = None
        
        conf_mat, loss = train_val_loop(model, X, y, X_test, y_test, 
            training_dict, log_)
        agg_conf_mat.append(conf_mat.cpu().numpy())
        agg_loss.append(loss.cpu().numpy())

        fold_num += 1
    
    agg_conf_mat = np.array(agg_conf_mat)
    agg_loss = np.array(agg_loss)

    return agg_conf_mat, agg_loss

def infer(X_tensor, y_tensor, model, loss_fn, device, perms=None, num_snps=0, 
            class_weights=[1,1], batch_size=256):
    """Helper function used to start model inference.

    Parameters:
    ----------
    X_tensor : numpy ndarray
        Training data.
    y_tensor : numpy ndarray
        Training labels.
    model : str
        Path to saved trained model.
    loss_fn : PyTorch loss object
        Loss function object. 
    device : str
        GPU/CPU name.
    class_weigths: list (nclasses,)
        Class weights used in calculating the metrics (default=[1,1]
        representing balanced binary classification)

    Returns
    -------
    y_pred : Numpy ndarray
        Network predictions.
    conf_mat : tuple of float
        (tn, fp, fn, tp)
    loss : float
        Loss.
    """
    if perms is None:
        perms = [torch.arange(0, X_tensor.shape[0]),]

        ##########################
        # ONLY FOR 3GroupTrainNN #
        ##########################
        # perms = [torch.arange(0, X_tensor.shape[0]*X_tensor.shape[1]),]

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
            #########################
            # ONLY FOR 1GroupTrainNN #
            #########################
            X_[:, :, :num_snps] = X_[perm, :, :num_snps]
            # X_ = X_[perm]

            dataloader = FastTensorDataLoader(X_, y_tensor, batch_size=batch_size)
            y_pred = torch.tensor([], device=device).long()
            loss = 0.0
            
            for bnum, sample in enumerate(dataloader):
                X_batch = sample[0]
                y_batch = sample[1]
                
                model.eval()
                model = model.to(device)
                # model.beta_matrix = model.beta_matrix.to(device)
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

def train(X, y, X_test, y_test, num_snps, model_dict, optim_dict, train_dict, 
        device):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    y : [type]
        [description]
    X_test : [type]
        [descri
        ption]
    y_test : [type]
        [description]
    num_snps : [type]
        [description]
    model_dict : [type]
        [description]
    optim_dict : [type]
        [description]
    train_dict : [type]
        [description]
    device : [type]
        [description]
    """
    log = train_dict['log']
    model_name = model_dict['model_name']
    model_type = model_dict['model_type']
    model_args = model_dict['model_args']
    
    lr = optim_dict['LR']
    damp = optim_dict['damping']
    class_weights = optim_dict['class_weights']
    class_weights = torch.Tensor(class_weights)
    optim = optim_dict['optim']

    # Train model on unpermuted data
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.07, 
        random_state=447)
    agg_conf_mat, agg_loss = start_training(
        X, y, X_test, y_test, splitter, train_val_loop, model_dict, 
        optim_dict, train_dict, device, 0)

    metric_logs = '{}/{}'.format(train_dict['log'], 'training_metrics.npz')
    np.savez(metric_logs, agg_conf_mat=agg_conf_mat, agg_loss=agg_loss)


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


# Experiment summary generation

def return_data(d, model_fold, label, bp, sys_params, covs, 
    train_ids_f, test_ids_f):
    geno = None
    phen_cov = [] 

    gs, model_path, Xs, ys, snps, cws, colss = [], [], [], [], [], [], [] 
    for i, gene in enumerate(d['names']):
        chrom = d['chrom'][i]
        X, y, X_test, y_test, class_weights, data_cols, num_snps = load_data(
            {chrom:geno}, phen_cov, [gene,], [chrom,], label, bp, 
            '/home/upamanyu/GWASOnSteroids/Runs', 
            sys_params, covs, train_ids_f, test_ids_f,
            over_coeff=0.0, 
            balance=1.0, 
            SNP_thresh=1000)
        # X_test, y_test = X_test[0], y_test[0]
        # X_test = np.concatenate((X_test[0], X_test[1]))
        # y_test = np.concatenate((y_test[0], y_test[1]))
        _, _, Xt, yt = group_data_prep(None, None, X_test, y_test, 10, covs)
        
        # skf = StratifiedKFold(n_splits=3, shuffle=False)
        # for train_index, test_index in skf.split(Xt, yt):
        #     Xt = Xt[test_index]
        #     yt = yt[test_index]
        #     break
        
        gs.append(gene)
        model_path.append('{}/{}/{}_{}.pt'.format(model_fold, gene, num_snps, gene))
        Xs.append(Xt)
        ys.append(yt)
        snps.append(num_snps)
        cws.append(class_weights)
        colss.append(data_cols)

    return gs, model_path, Xs, ys, snps, cws, colss

def return_stats(df, unc_p, cor_p):
    """[summary]

    Parameters
    ----------
    df : [type]
        [description]
    unc_p : [type]
        [description]
    cor_p : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    tot = df.shape[0]
    cor = df.loc[df['P_Acc'] <= cor_p].shape[0]
    unc = df.loc[df['P_Acc'] <= unc_p].shape[0]
    vprint('Corrected: {}/{} = {:.2f}'.format(cor, tot, 100*cor/tot))
    vprint('Uncorrected: {}/{} = {:.2f}\n'.format(unc, tot, 100*unc/tot))
    return tot, cor, unc

def ptest_neg_summary_stats(logs_file, cor_p):
    """[summary]

    Parameters
    ----------
    logs_file : [type]
        [description]
    cor_p : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    df = pd.read_csv(logs_file)
    df.set_index('Gene', drop=False, inplace=True)
    with open('./params/gene_subsets.yaml', 'r') as f:
        g_yaml = yaml.load(f, Loader=yaml.FullLoader)

    summary_dict = {
        'Unc_Neg':0,
        'Corr_Neg':0
    }
    
    vprint('OVERALL (NEG)')
    vprint('------------')
    t, c, u = return_stats(df, 0.05, cor_p)
    summary_dict['Unc_Neg'] = u
    summary_dict['Corr_Neg'] = c
    
    try:
        df11 = df.loc[g_yaml['First_GroupTrain_Hits_Neg']['names']]
        vprint('OLD 11 (NEG)')
        vprint('------------')
        t, c, u = return_stats(df11, 0.05, cor_p)
        summary_dict['Unc_11_Neg'] = u
        summary_dict['Corr_11_Neg'] = c
        
        dfM = df.loc[~df.index.isin(df11.index.tolist())]
        vprint('MARIONI (NEG)')
        vprint('------------')
        t, c, u = return_stats(dfM, 0.05, cor_p)
        summary_dict['Unc_Marioni_Neg'] = u
        summary_dict['Corr_Marioni_Neg'] = c
    except:
        print('Diabetes')

    return summary_dict

def ptest_pos_summary_stats(logs_file, cor_p):
    """[summary]

    Parameters
    ----------
    logs_file : [type]
        [description]
    cor_p : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    df = pd.read_csv(logs_file)
    df.set_index('Gene', drop=False, inplace=True)

    with open('./params/gene_subsets.yaml', 'r') as f:
        g_yaml = yaml.load(f, Loader=yaml.FullLoader)

    summary_dict = {
        'Unc_Pos':0,
        'Corr_Pos':0
    }
    
    vprint('OVERALL (POS)')
    vprint('------------')
    t, c, u = return_stats(df, 0.05, cor_p)
    summary_dict['Unc_Pos'] = u
    summary_dict['Corr_Pos'] = c
    try:
        df19 = df.loc[g_yaml['First_GroupTrain_Hits_Pos']['names']]
        vprint('OLD 19 (POS)')
        vprint('------------')
        t, c, u = return_stats(df19, 0.05, cor_p)
        summary_dict['Unc_19_Pos'] = u
        summary_dict['Corr_19_Pos'] = c
        
        dfM = df.loc[df.index.isin(g_yaml['Marioni_Top50']['names'])]
        vprint('MARIONI (POS)')
        vprint('------------')
        t, c, u = return_stats(dfM, 0.05, cor_p)
        summary_dict['Unc_Marioni_Pos'] = u
        summary_dict['Corr_Marioni_Pos'] = c
        
        dfK = df.loc[~df.index.isin(dfM.index.tolist() + df19.index.tolist())]
        vprint('Kegg (POS)')
        vprint('------------')
        t, c, u = return_stats(dfK, 0.05, cor_p)
        summary_dict['Unc_KEGG_Pos'] = u
        summary_dict['Corr_KEGG_Pos'] = c
    except:
        print('Diabetes')

    return summary_dict

def method_comparison(df, cor_p, fname):
    x = df.Gene.values
    y = df.columns[1:]
    data = df.iloc[:, 1:].values <= cor_p
    data = data.T
    data = np.repeat(data, 2, axis=1)
    data = np.repeat(data, 5, axis=0)
    cmap = plt.get_cmap('bwr')
    cmap.set_under('k', alpha=0)
    plt.imshow(data, cmap=cmap, vmin=0.5)
    plt.xticks(np.arange(len(x))*2, x, fontsize=2, rotation=90)
    plt.yticks(np.arange(len(y))*5+2, y, fontsize=4)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def acc_vs_P(dfs, cor_p, fpath, exp_name):
    labels = ['Pos', 'Neg', 'Rand']
    colors = ['blue', 'red', 'green']
    markers = ['o', '^', 'x']
    # labels = ['Na0', 'LayerNorm', 'Vanilla']
    # colors = ['blue', 'red', 'green']
    # markers = ['o', '^', 'x']
    xs = []
    data = []
    lines = []
    zs = []
    min_x, max_x = 10000, -10000
    for i, df in enumerate(dfs):
        xs.append(df['Acc'].values)
        min_x = min(min_x, np.min(xs[-1]))
        max_x = max(max_x, np.max(xs[-1]))
        data.append(-np.log10(df['P_Acc'].values))
        if i == 1:
            lines.append(plt.scatter(xs[-1], data[-1], label=labels[i], alpha=1, 
                s=3, marker=markers[i], c=colors[i]))
        else:
            lines.append(plt.scatter(xs[-1], data[-1], label=labels[i], alpha=0.5, 
                s=3, marker=markers[i], c=colors[i]))
        zs.append(np.polyfit(xs[-1], data[-1], 1))
    
    # x_ = np.linspace(min_x, max_x, 1000)
    # for i, z in enumerate(zs):
    #     plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
    #         c=lines[i].get_facecolors()[0])
    
    plt.gca().axhline(-np.log10(0.05), linestyle='--', alpha=0.5, 
        color='k', linewidth=0.5)
    plt.gca().axhline(-np.log10(cor_p), linestyle='--', alpha=0.5, 
        color='green', linewidth=0.5)
    # plt.gca().axvline(0.62, linestyle='--', alpha=0.5, 
    #     color='orange', linewidth=0.5)
    # plt.gca().axvline(0.5874, linestyle='--', alpha=0.5, 
    #     color='green', linewidth=0.5)
    # plt.gca().axvspan(0.5874, max(xs[0]+0.0005), 
    #     -np.log10(cor_p)/max(data[-1]) + 0.025, 1,
    #     alpha=0.2, facecolor='green', zorder=-1)
    
    plt.xlabel('Acc')
    plt.ylabel('-log10 (P)')
    plt.title('{} - Acc vs P'.format(exp_name))
    plt.legend()
    # fname = '{}_Acc_vs_P.svg'.format(fpath)
    fname = '{}_Acc_vs_P.png'.format(fpath)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def SNPs_vs_P(dfs, cor_p, fpath, exp_name):
    labels = ['Pos', 'Neg', 'Rand']
    colors = ['blue', 'red', 'green']
    markers = ['o', '^', 'x']

    xs = []
    data = []
    lines = []
    num_snps = dfs[0]['SNPs'].values
    xs = np.quantile(num_snps, np.arange(1, 11, 1)*0.1)
    xs = np.array(xs, dtype=int)
    print(xs)
    xs_ = []
    snps = []
    for i, df in enumerate(dfs):
        prev_x = 0
        for j, x in enumerate(xs):
            interval = x - prev_x
            expansion = 50/interval

            tdf = df.loc[(df['SNPs'] > prev_x) & (df['SNPs'] <= x)]
            data.append(-np.log10(tdf['P_Acc'].values))
            # xs_.extend(np.repeat(x, len(data[-1])))
            xs_.extend(tdf['SNPs'].values)
            # snps.extend((50*j)+(tdf['SNPs'].values - prev_x)*expansion)
            snps.extend(tdf['SNPs'].values)
            prev_x = x
            # print(round(min(snps)), round(max(snps)))
            # print()
        
        snps = np.array(snps) 
        # snps = -np.log(snps/max(snps))
        xs_ = snps
        data = np.concatenate(data)
        lines.append(plt.scatter(snps, data, color=colors[i], alpha=0.5,
                marker=markers[i], label=labels[i], s=4))
        z = np.polyfit(xs_, data, 1)
        x_ = np.linspace(min(xs_), max(xs_), 1000)
        # plt.plot(x_*(len(xs)*50/max(xs_)), np.poly1d(z)(x_), ':', alpha=0.5, 
        #     c=lines[i].get_facecolors()[0])
        plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
            c='r')
        snps = [] 
        data = []
        xs_ = []
    
    plt.gca().axhline(-np.log10(0.05), linestyle='--', alpha=0.3, 
        color='k', linewidth=0.5)
    plt.gca().axhline(-np.log10(cor_p), linestyle='--', alpha=0.3, 
        color='r', linewidth=0.5)
    # plt.xlabel('Num SNPs (ln)')
    plt.xlabel('Num SNPs')
    plt.ylabel('-log10 (P_Acc)')
    plt.title('{} - SNPs vs P'.format(exp_name))
    # print(xs)
    # plt.xticks(np.arange(50, (len(xs)+1)*50, 50), np.around(xs, 0), fontsize=4)
    # plt.xticks(np.arange(50, (len(xs)+1)*50, 50), np.around(xs, 0), fontsize=4)
    plt.legend()
    # fname = '{}_SNPs_vs_P.svg'.format(fpath)
    fname = '{}_SNPs_vs_P.png'.format(fpath)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def SNPs_vs_Acc(dfs, cor_p, fpath, exp_name):
    labels = ['Pos', 'Neg', 'Rand']
    colors = ['blue', 'red', 'green']
    markers = ['o', '^', 'x']
    # labels = ['Na0', 'LayerNorm', 'Vanilla']
    # colors = ['blue', 'red', 'green']
    # markers = ['o', '^', 'x']
    xs = []
    data = []
    lines = []
    num_snps = dfs[0]['SNPs'].values
    xs = np.array(xs, dtype=int)
    xs = np.quantile(num_snps, np.arange(1, 11, 1)*0.1)
    xs_ = []
    snps = []
    for i, df in enumerate(dfs):
        prev_x = 0
        for j, x in enumerate(xs):
            interval = x - prev_x
            expansion = 50/interval

            tdf = df.loc[(df['SNPs'] > prev_x) & (df['SNPs'] <= x)]
            data.append(tdf['Acc'].values)
            xs_.extend(np.repeat(x, len(data[-1])))
            # snps.extend((50*j)+(tdf['SNPs'].values - prev_x)*expansion)
            snps.extend(tdf['SNPs'].values)
            prev_x = x
            # print(round(min(snps)), round(max(snps)))
            # print()
        
        snps = np.array(snps) 
        # snps = np.log10(snps/max(snps))
        xs_ = snps

        data = np.concatenate(data)
        lines.append(plt.scatter(snps, data, color=colors[i], alpha=0.5,
                marker=markers[i], label=labels[i], s=4))
        z = np.polyfit(xs_, data, 1)
        x_ = np.linspace(min(xs_), max(xs_), 1000)
        # plt.plot(x_*(len(xs)*50/max(xs_)), np.poly1d(z)(x_), ':', alpha=0.5, 
        #     c=lines[i].get_facecolors()[0])
        plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
            c='r')
        snps = [] 
        data = []
        xs_ = []
     
    # plt.xlabel('Num SNPs (log10)')
    plt.xlabel('Num SNPs')
    plt.ylabel('Acc')
    plt.title('{} - SNPs vs Acc'.format(exp_name))
    # plt.xticks(np.arange(50, (len(xs)+1)*50, 50), np.around(xs, 0), fontsize=4)
    plt.legend()
    # fname = '{}_SNPs_vs_Acc.svg'.format(fpath)
    fname = '{}_SNPs_vs_Acc.png'.format(fpath)
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    
    # labels = ['Pos', 'Neg']
    # colors = ['blue', 'red']
    # markers = ['o', '^']
    # xs = []
    # data = []
    # lines = []
    # zs = []
    # min_x, max_x = 10000, 0
    # for i, df in enumerate(dfs):
    #     xs.append(np.log10(df['SNPs'].values))
    #     min_x = min(min_x, np.min(xs[-1]))
    #     max_x = max(max_x, np.max(xs[-1]))
    #     data.append(df['Acc'].values)
    #     lines.append(plt.scatter(xs[-1], data[-1], label=labels[i], alpha=0.8, 
    #         s=3, marker=markers[i], c=colors[i]))
    #     zs.append(np.polyfit(xs[-1], data[-1], 1))

    # x_ = np.linspace(min_x, max_x, 1000)
    # for i, z in enumerate(zs):
    #     plt.plot(x_, np.poly1d(z)(x_), ':', alpha=0.5, 
    #     c=lines[i].get_facecolors()[0])

    # plt.xlabel('log10 (Num_SNPs)')
    # plt.ylabel('Acc')
    # plt.title('{} - SNPs_vs_Acc.svg'.format(fpath))
    # plt.legend()
    # fname = '{}_SNPs_vs_Acc.svg'.format(fpath)
    # plt.savefig(fname, bbox_inches='tight')
    # plt.close()

def manhattan(dfs, cor_p, fpath, exp_name, genes_df):
    plt.figure(figsize=(10,7))

    xs = []
    data = []
    lines = []
    chrom_size_dict = {}
    prev = 0
    ticks = np.arange(1, 23)
    
    for c in range(1, 23):
        temp = [0,0] 
        temp[0] = prev
        temp[1] = prev + (np.max(genes_df.loc[genes_df['chrom'] == str(c)]['end']))/(10**7)
        prev = temp[1]
        chrom_size_dict[c] = temp
        ticks[c-1] = np.mean(temp)

    min_x = 0
    max_x = chrom_size_dict[22][1] + 1

    # colors = ['blue', 'green', 'red']
    # markers = ['o', '^', 'x']
    colors = np.tile(['mediumblue', 'deepskyblue'], 11)
    markers = np.repeat(['o'], 22)
    texts = []
    for i, df in enumerate(dfs):
        pos = genes_df.loc[df['Gene']]['start'].values
        pos = [chrom_size_dict[c][0] for c in df['Chrom'].values] + pos/(10**7)        
        x = pos
        xs.append(x)
        data.append(-np.log10(df['P_Acc'].values))
        lines.append(plt.scatter(xs[-1], data[-1], alpha=0.5, s=5, 
            marker=markers[i], color=colors[i]))
        for ind in range(len(xs[-1])):
            if df.iloc[ind]['Gene'] in ['PIGK', 'NRXN1', 'LRP1B', 'LYPD6B', 'PARD3B', 'RBMS3', 'STAC', 'FLNB', 'ROBO1', 'CADM2', 'EPHA6', 'CHCHD6', 'ATP1B3', 'SERPINI1', 'EFNA5', 'IMMP2L', 'EPHA1-AS1', 'CTNNA3', 'CNTN5', 'PITPNM2', 'GPR137C', 'TMEM170A', 'TOMM40', 'BCAM', 'APOC1', 'PPP1R37', 'EXOC3L2']:
                continue
            if data[-1][ind] >= -np.log10(cor_p):
                texts.append(plt.text(xs[-1][ind], data[-1][ind], df.iloc[ind]['Gene'], 
                    fontdict={'size':6}, rotation=90))
    adjust_text(texts, force_text=(0.4, 0.5), force_points=(0.5, 0.8),
                    arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    # plt.axhline(-np.log10(0.05), linestyles='--', alpha=0.5, colors='k', linewidth=0.5)
    plt.axhline(-np.log10(1e-5), linestyle='--', alpha=0.5, color='k', linewidth=0.5)
    plt.axhline(-np.log10(cor_p), linestyle='--', alpha=0.5, color='r', linewidth=0.5)
    plt.yticks(fontsize=14)
    # plt.ylim((0, max(np.concatenate(data))))
    plt.xticks(ticks, np.arange(1, 23), fontsize=14, rotation=90)
    plt.grid(axis='x', alpha=0.3)
    plt.xlabel('Chrom', fontsize=14)
    plt.ylabel('-log10 (P)', fontsize=14)
    plt.title('{} - manhattan'.format(exp_name))
    
    fname = '{}_manhattan.svg'.format(fpath)
    # fname = '{}_manhattan.png'.format(fpath)
    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()

def acc_compare(dfs, cor_p, fpath, exp_name, exp_logs):
    for t, elogs, df in zip(['Pos', 'Neg'], exp_logs, dfs):
        print(t, elogs)
        fig, ax = plt.subplots(3, 1)
        ax = ax.flatten()
        data = [[],[],[]]
        gs = [[], [], []]
        for g in df['Gene'].to_list():
            print(g)
            if not os.path.isdir(os.path.join(elogs, g)):
                continue
            perm_a = np.load('{}/{}/ptest_metrics.npz'.format(elogs, g))['acc']
            p = df.loc[g]['P_Acc']
            # a = perm_a
            a = [perm_a[0],] + list(np.round(np.random.choice(perm_a, 999, replace=False), 2))
            if p <= cor_p:
                data[0].append(a)
                gs[0].append(g)
            elif p <= 0.05 and p > cor_p:
                data[1].append(a)
                gs[1].append(g)
            else:
                data[2].append(a)
                gs[2].append(g)
            
        for i in range(len(data)):
            print(i)
            d = np.asarray(data[i])
            print(d.shape)
            ax[i].violinplot(d.T)
            ax[i].scatter(np.arange(1, len(d)+1), d[:, 0], c='r', s=3, 
                marker='x')
            ax[i].set_xticks(np.arange(1, len(d)+1))
            ax[i].set_xticklabels(gs[i], fontsize=6, rotation=90)
            ax[i].grid(True, axis='x')
            ax[i].set_ylabel('Accuracy')
            if i == 0:
                ax[i].set_title('P <= {:3f}'.format(cor_p))
            elif i == 1:
                ax[i].set_title('P in [0.05, {:3f})'.format(cor_p))
            else:
                ax[i].set_title('P > 0.05')
        
        fig.tight_layout()
        fig.savefig('{}_{}_Acc_compare.svg'.format(fpath, t))
        plt.close()

def overfit_ratio(dfs, cor_p, fpath, exp_name, exp_logs):

    for t, elogs, df in zip(['Pos', 'Neg'], exp_logs, dfs):
        fig, ax = plt.subplots(4, 3, sharex=True)
        # ax = ax.flatten()
        data = []
        gs = [] 
        c = ['g', 'b', 'k', 'r']
        for g in df['Gene'].to_list():
            gp = os.path.join(elogs, g)
            if not os.path.isdir(gp):
                continue
            
            metrics = np.load('{}/{}/training_metrics.npz'.format(elogs, g))
            cm_train = metrics['agg_conf_mat'][0][:, 0]
            cm_val = metrics['agg_conf_mat'][0][:, 1]
            a_train = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_train])
            a_val = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_val])
            
            over_ratio = a_val/a_train
            mean_or = np.mean([np.mean(over_ratio[i*100:(i+1)*100]) for i in range(0,5)])
            mean_or = np.mean([np.mean(over_ratio[i*100:(i+1)*100]) for i in range(0,5)])
            
            if mean_or >= 1:
                i = 0
            elif mean_or < 1 and mean_or >= 0.9:
                i = 1
            elif mean_or < 0.9 and mean_or >= 0.8:
                i = 2
            else:
                i = 3        
            
            if df.loc[g]['P_Acc'] <= cor_p:
                ax[i, 0].plot(over_ratio, c=c[i], linewidth=0.5, alpha=0.5, 
                    linestyle='-')
                # ax[i, 1].text(len(over_ratio)-1, over_ratio[-1], '{}'.format(g), 
                #     fontdict=dict(fontsize=2))
            if df.loc[g]['P_Acc'] > cor_p and df.loc[g]['P_Acc'] <= 0.05:
                ax[i, 1].plot(over_ratio, c=c[i], linewidth=0.5, alpha=0.5, 
                    linestyle='-')
            else:
                ax[i, 2].plot(over_ratio, c=c[i], linewidth=0.5, alpha=0.3, 
                    linestyle=':')
            gs.append(g)

        ax[0, 0].set_title('P <= {:3f}'.format(cor_p))
        ax[0, 1].set_title('P in [0.05, {:3f})'.format(cor_p))
        ax[0, 2].set_title('P > 0.05')

        fig.tight_layout()
        fig.savefig('{}_{}_Overfit_ratio.svg'.format(fpath, t))
        plt.close()

def acc_overfit_ratio(dfs, cor_p, fpath, exp_name, exp_logs):
    for t, elogs, df in zip(['Pos', 'Neg'], exp_logs, dfs):
        data = [[],[],[]]

        data2 = [[],[],[]]
        y_off = [1, 1, 1]
        gs = [[], [], []] 
        c = ['g', 'b', 'k']
        for g in df['Gene'].to_list():
            gp = os.path.join(elogs, df.loc[g]['Gene_win'])
            if not os.path.isdir(gp):
                continue
            
            p = df.loc[g]['P_Acc']
            perm_a = np.load('{}/{}/ptest_metrics.npz'.format(elogs, df.loc[g]['Gene_win']))['acc']
            # a = perm_a
            a = [perm_a[0],] + list(np.round(np.random.choice(perm_a, 999, replace=False), 2))
            
            metrics = np.load('{}/{}/training_metrics.npz'.format(elogs, df.loc[g]['Gene_win']))
            cm_train = metrics['agg_conf_mat'][0][:, 0]
            cm_val = metrics['agg_conf_mat'][0][:, 1]
            a_train = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_train])
            a_val = np.asarray([metrics_from_conf_mat(x)[3] for x in cm_val])
            
            o_r = a_val
            # /a_train
            if p <= cor_p:
                ind = 0
            elif p > cor_p and p <= 0.05:
                ind = 1
            else:
                ind = 2

            data[ind].append(a)
            gs[ind].append(g)
            o_r = (o_r - min(o_r))/(max(o_r) - min(o_r))
            o_r = (2*o_r) + (y_off[ind]-1)
            data2[ind].append(o_r)
            y_off[ind] += 2

        fig, ax = plt.subplots(3, 2, figsize=(10, 20), 
            gridspec_kw={'width_ratios': [1, 1], 
                        'height_ratios': [len(data[0])/len(df), 
                                len(data[1])/len(df), 
                                len(data[2])/len(df)]})

        for i in range(len(data)):
            # print(gs[i])
            d = np.asarray(data[i])
            d2 = np.asarray(data2[i])
            if len(d) == 0:
                continue
            [ax[i, 1].plot(d2_j, c=c[i], linewidth=0.5) for d2_j in d2]
            vp = ax[i, 0].violinplot(d.T, positions=np.arange(1, y_off[i], 2),
                vert=False, widths=2)
            vp['cmins'].set_linewidth(0)
            vp['cmaxes'].set_linewidth(0)
            vp['cbars'].set_linewidth(0.7)

            ax[i, 0].scatter(d[:, 0], np.arange(1, y_off[i], 2), c='r', s=5, 
                marker='x')
            ax[i, 0].scatter(1-d[:, 0], np.arange(1, y_off[i], 2), c='k', s=5, 
                marker='x', alpha=0.3)
            
            ax[i, 0].axvline(0.5, color='k', linestyle=':', linewidth=0.5, 
                alpha=0.6)
            ax[i, 0].set_xlim((0, 1))
            ax[i, 0].set_ylim((0, y_off[i]+1))
            ax[i, 0].set_yticks(np.arange(1, y_off[i], 2))
            ax[i, 0].set_yticklabels(gs[i], fontsize=4)
            ax[i, 0].grid(axis='y', color='r', linestyle=':', linewidth=0.5)
            
            ax[i, 1].set_ylim((0,y_off[i]+1))
            ax[i, 1].set_yticks(np.arange(1, y_off[i], 2))
            ax[i, 1].set_yticklabels([])
            ax[i, 1].grid(axis='y', color='r', linestyle=':', linewidth=0.5)

            if i == 0:
                ax[i, 0].set_ylabel('P <= {:3f}'.format(cor_p), fontsize=6)
            elif i == 1:
                ax[i, 0].set_ylabel('P in [0.05, {:3f})'.format(cor_p), fontsize=6)
            else:
                ax[i, 0].set_ylabel('P > 0.05', fontsize=8)

        ax[2, 0].set_xlabel('Accuracy')
        ax[2, 1].set_xlabel('Epochs')
        fig.tight_layout()
        fig.savefig('{}_{}_acc_overfit.svg'.format(fpath, t))
        plt.close()

def summary_plot(df, fname, totals=[]):
    if len(totals) == 0:
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax = ax.flatten()
        if len(df.columns) > 5:
            totals = np.asarray([114, 114, 49, 49, 19, 19, 50, 50, 61, 61, 50, 50, 11, 11])
            confints = proportion_confint(df.iloc[:, 1:], totals, method='beta')
            df.iloc[:, 1:] = df.iloc[:, 1:]/totals
            df = df.iloc[:, 1:]

            x = np.arange(1, len(df)+1)*2
            ls = {'Unc':':', 'Corr':'-'}
            sym = {'Unc':'x', 'Corr':'o'}
            colours = {'Marioni':{'Neg':'red', 'Pos':'blue'},
                'KEGG':{'Neg':'orange', 'Pos':'green'},
                '11':{'Neg':'magenta'}, '19':{'Pos':'black'}}

            eps = {'Marioni':{'Neg':0.6, 'Pos':0},
                'KEGG':{'Neg':0.2, 'Pos':0.2},
                '11':{'Neg':0.6}, '19':{'Pos':0}}
            for i, c in enumerate(df.columns):
                try:
                    s, g, pn = c.split('_')
                except ValueError:
                    continue
                col = colours[g][pn]
                e = eps[g][pn]
                data = np.stack((confints[0][df.columns.to_list().index(c)], 
                    df[c].values, confints[1][df.columns.to_list().index(c)]), axis=-1)
                
                ind = 0
                if 'Marioni' in c or 'KEGG' in c:
                    if s == 'Unc':
                        ind = 0
                    else:
                        ind = 1
                else:
                    if s == 'Unc':
                        ind = 2
                    else:
                        ind = 3
                    
                ax[ind].boxplot(x=data.T, positions=x+e, showbox=False, widths=0,
                    medianprops={'color':col, 'marker':sym[s], 'markersize':2},
                    capprops={'color':col, 'marker':'_', 'markersize':2, 'alpha':0.3},
                    whiskerprops={'color':col, 'linewidth':0.5, 'alpha':0.3},
                    whis=0, showcaps=False, showfliers=False)
                # ax[1].scatter(x, df[c], label=c, c=col, marker=sym[s],
                    # s=9, alpha=0.5)
            ax[0].set_title('Marioni and KEGG (Unc)')
            ax[1].set_title('Marioni and KEGG (Corr)')
            ax[2].set_title('Old Group Train hits (Unc)')
            ax[3].set_title('Old Group Train hits (Corr)')
            # ax[0].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
            # ax[1].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
            for a in ax:
                a.axhline(0.05, linestyle=':', c='k')
                a.set_ylim(top=1)
                a.set_xticks([0,]+list(x))
                a.set_xticklabels(['',]+df.index.to_list(), dict(fontsize=6, rotation=90))
                
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

        else:
            fig, ax = plt.subplots(1, 2)
            ax = ax.flatten()
            totals = np.asarray([120, 120, 120, 120])
            confints = proportion_confint(df.iloc[:, 1:], totals, method='beta')
            df.iloc[:, 1:] = df.iloc[:, 1:]/totals
            df = df.iloc[:, 1:]

            x = np.arange(1, len(df)+1)*2
            ls = {'Unc':':', 'Corr':'-'}
            sym = {'Unc':'x', 'Corr':'o'}
            colours = {'Neg':'red', 'Pos':'blue'}
    
            for i, c in enumerate(df.columns):
                try:
                    s, pn = c.split('_')
                except ValueError:
                    continue
                col = colours[pn]
                data = np.stack((confints[0][df.columns.to_list().index(c)], 
                    df[c].values, confints[1][df.columns.to_list().index(c)]), axis=-1)
                ind = 0
                if s == 'Unc':
                    ind = 0
                else:
                    ind = 1
                ax[ind].boxplot(x=data.T, positions=x, showbox=False, widths=0,
                    medianprops={'color':col, 'marker':sym[s], 'markersize':2},
                    capprops={'color':col, 'marker':'_', 'markersize':2, 'alpha':0.3},
                    whiskerprops={'color':col, 'linewidth':0.5, 'alpha':0.3},
                    whis=0, showcaps=False, showfliers=False)
            ax[0].set_title('Unc')
            ax[1].set_title('Corr')
            for a in ax:
                a.axhline(0.05, linestyle=':', c='k')
                a.set_ylim(top=1)
                a.set_xticks([0,]+list(x))
                a.set_xticklabels(['',]+df.index.to_list(), dict(fontsize=6, rotation=90))
                
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

def binomial_plot(dfs, fname):
    
    dfs = sort_values
    
    
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax = ax.flatten()
    totals = np.asarray([114, 114, 49, 49, 19, 19, 50, 50, 61, 61, 50, 50, 11, 11])
    df.iloc[:, 1:] = 100*(df.iloc[:, 1:]/totals)
    df = df.iloc[:, 1:]

    x = np.arange(len(df))
    ls = {'Unc':':', 'Corr':'-'}
    sym = {'Unc':'x', 'Corr':'o'}
    colours = {'Marioni':{'Neg':'red', 'Pos':'blue'},
        'KEGG':{'Neg':'orange', 'Pos':'green'},
        '11':{'Neg':'magenta'}, '19':{'Pos':'black'}}
                
    for i, c in enumerate(df.columns):
        try:
            s, g, pn = c.split('_')
        except ValueError:
            continue
        col = colours[g][pn]

        confint = proportion_confint(df[c], 
            totals[df.columns.to_list().index(c)], method='beta')
        data = np.stack((confint[0], df[c].values, confint[1]), axis=-1)
        if 'Marioni' in c or 'KEGG' in c:
            # ax[0].boxplot(data, positions=x, showbox=False, widths=0,
            #     medianprops={'color':col, 'marker':sym[s], 's':9}, label=c)
            ax[0].scatter(x, df[c], label=c, c=col, marker=sym[s], s=9, alpha=0.5)
        else:
            # ax[1].boxplot(data, positions=x, showbox=False, widths=0,
            #     medianprops={'color':col, 'marker':sym[s], 's':9}, label=c)
            ax[1].scatter(x, df[c], label=c, c=col, marker=sym[s], s=9, alpha=0.5)

    ax[0].set_title('Marioni and KEGG')
    ax[1].set_title('Old Group Train hits')
    ax[0].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
    ax[1].legend(bbox_to_anchor=(1 ,1), loc='upper left', fontsize=7)
    ax[0].axhline(0.05, linestyle=':', c='k')
    ax[1].axhline(0.05, linestyle=':', c='k')
    plt.xticks(x, df.index.values, fontsize=6, rotation=90)
    plt.savefig(fname)
    plt.close()
    #'beta' produces same results as R
    proportion_confint(method='beta') 

def gen_summary_documents(exp_base, exp_name, docs='1111'):
    with open('./params/system_specific_params.yaml', 'r') as params_file:
        sys_params = yaml.load(params_file, Loader=yaml.FullLoader)
    base_fold = sys_params['SUMMARY_BASE_FOLDER']
    exp_fold = '{}/{}'.format(base_fold, exp_name)
 
    genes_df = pd.read_csv('/home/upamanyu/GWASOnSteroids/GWASNN/datatables/genes.csv')
    genes_df.drop_duplicates(['symbol'], inplace=True)
    genes_df.set_index('symbol', drop=False, inplace=True)
    
    if not os.path.isdir(exp_fold):
        os.mkdir(exp_fold)

    cor_p = 3.397431541754434e-06

    # 1. Log Pos/Neg Corr+Uncorr ratios
    if int(docs[0]):
        summary_file = '{}/exp_summaries.csv'.format(base_fold)
        sd = {'Exp':exp_name}
        sd.update(ptest_pos_summary_stats(exp_base.format('Pos', 'Pos'), cor_p))
        sd.update(ptest_neg_summary_stats(exp_base.format('Neg', 'Neg'), cor_p))
        if os.path.isfile(summary_file):
            summary_df = pd.read_csv(summary_file)
            summary_df.set_index('Exp', drop=False, inplace=True)
            if exp_name not in summary_df.index.values:
                summary_df = summary_df.append(pd.DataFrame(sd, index=[exp_name,]))
                summary_df.to_csv(summary_file, index=False)
            else:
                print(exp_name, ' exists in summary file.')
        else:
            summary_df = pd.DataFrame(sd, index=[exp_name,])
            summary_df.to_csv(summary_file, index=False)
        summary_plot(summary_df, summary_file.replace('csv', 'svg'))
    
    # 2. Generate gradient plots (Diff plot)
    if int(docs[1]):
        
        for t in ['Pos', 'Neg']:
            fig_name = '{}/{}_{}_grads.svg'.format(exp_fold, t, exp_name)
            if os.path.isfile(fig_name):
                print(fig_name, ' exists')
                continue
            p_file = exp_base.format(t, t)
            model_fold = '/'.join(p_file.split('/')[:-1])
            df = pd.read_csv(p_file)
            
            df.sort_values('P_Acc', inplace=True)
            g_df = genes_df.loc[df.Gene.tolist()]
            g_df['SNPs'] = pd.Series()
            g_df.loc[df.Gene.tolist(), 'Perms'] = df['Perms'].values
            g_df.loc[df.Gene.tolist(), 'SNPs'] = df['SNPs'].values
            g_df.loc[df.Gene.tolist(), 'P_Acc'] = df['P_Acc'].values
            g_df.loc[df.Gene.tolist(), 'Type'] = df['Type'].values

            ds = []
            num_procs = 10
            d_size = (len(g_df)//num_procs)
            for i in range(0, num_procs):
                start = i*d_size
                end = (i+1)*d_size if i != num_procs-1 else len(g_df)
                d = {
                    'names': g_df.symbol.tolist()[start:end],
                    'chrom': g_df.chrom.tolist()[start:end],
                    'ids': g_df.id.tolist()[start:end]
                }
                ds.append([d, model_fold])
            gs, model_path, Xs, ys, snps, cws, colss = [], [], [], [], [], [], []
            with mp.Pool(num_procs) as pool:
                res = pool.starmap(return_data, ds)
                pool.close()
                for r in res:
                    gs.extend(r[0])
                    model_path.extend(r[1])
                    Xs.extend(r[2])
                    ys.extend(r[3])
                    snps.extend(r[4])
                    cws.extend(r[5])
                    colss.extend(r[6])
            num_covs = len([c for c in colss[0] if 'rs' not in c])
            gradient_plot(gs, model_path, Xs, ys, snps, cws, colss, num_covs,
                torch.device('cuda:3'), exp_fold, '{}_{}'.format(t, exp_name))
            
            # gradient_pair_plot(gs, g_df['P_Acc'].values, model_path, Xs, ys, 
            #     snps, cws, colss, num_covs, torch.device('cuda:3'), exp_fold, 
            #     '{}_{}'.format(t, exp_name))
            
            # Model structure graph
            if not os.path.isfile('{}/{}.png'.format(exp_fold, exp_name)):
                model = torch.load(model_path[0], map_location=torch.device('cpu'))
                raw_out = model.forward(torch.from_numpy(Xs[0]).float())
                make_dot(raw_out).render('{}/{}'.format(exp_fold, exp_name), 
                    format='png')
    
    # 3. Update hit_comparison plots (Diff plots for Neg and Pos)
    if int(docs[2]):
        
        for t in ['Pos', 'Neg']:
            p_file = exp_base.format(t, t)
            df = pd.read_csv(p_file)
            df.sort_values('Chrom', inplace=True)
            hit_comp_file = '{}/{}_hit_comparison.csv'.format(base_fold, t)
            if os.path.isfile(hit_comp_file):
                hits_df = pd.read_csv(hit_comp_file)
                hits_df.set_index('Gene', drop=False, inplace=True)
                hits_df.loc[df.Gene.values, exp_name] = df['P_Acc'].values
                hits_df.to_csv(hit_comp_file, index=False)
            else:
                hits_df = df[['Gene', 'P_Acc']]
                hits_df.rename(columns={'P_Acc':exp_name}, inplace=True)
                hits_df.to_csv(hit_comp_file, index=False)
            fig_name = hit_comp_file.replace('csv', 'svg')
            method_comparison(hits_df, cor_p, fig_name)

    # 4. Generate all summary plots (Single plot for Neg and Pos)
    if int(docs[3]):
        
        dfs = []
        exp_logs = []
        for t in ['Pos']:#, 'Neg', 'Rand']:
            p_file = exp_base.format(t, t)
            try:
                df = pd.read_csv(p_file)
            except FileNotFoundError:
                print('{} Does not exist'.format(p_file))
                continue
            df.set_index('Gene', drop=False, inplace=True)
            df = df.loc[~df['Gene'].isin(['APOE', 'TOMM40', 'APOC1'])]
            df['P_Acc'] = 10**(-1*df['P_Acc'].values)
            dfs.append(df)
            exp_logs.append('/'.join(p_file.split('/')[:-1]))
        fig_path = '{}/{}'.format(exp_fold, exp_name)

        dfs_c = [df.sort_values('Chrom') for df in dfs]
        # manhattan(dfs_c, cor_p, fig_path, exp_name, genes_df)
        SNPs_vs_P(dfs_c, cor_p, fig_path, exp_name)
        SNPs_vs_Acc(dfs_c, cor_p, fig_path, exp_name)
        acc_vs_P(dfs_c, cor_p, fig_path, exp_name)
        
        dfs_p = [df.sort_values('P_Acc') for df in dfs]
        # acc_compare(dfs_p, cor_p, fig_path, exp_name, exp_logs)
        # overfit_ratio(dfs_p, cor_p, fig_path, exp_name, exp_logs)
        # acc_overfit_ratio(dfs_p, cor_p, fig_path, exp_name, exp_logs)

# Functions to help understand the results better

def generate_GRS(gene_dict, summary_df, label, bp, beta_header='b'):

    maf_df = pd.read_csv('/mnt/sdb/Summary_stats/MAFs.csv')
    maf_df.drop_duplicates(subset=['SNP'], inplace=True)
    maf_df.set_index('SNP', inplace=True)
    
    test_df = pd.read_csv('./params/test_ids.csv', dtype={'iid':int})
    # train_df = pd.read_csv('./params/train_ids.csv', dtype={'iid':int})
    # test_df = pd.concat((test_df, train_df))
    
    test_df.drop_duplicates(['iid'], inplace=True)
    test_ids = test_df['iid'].values

    agg_gtypes = pd.DataFrame(columns=['iid', 'label'])
    agg_gtypes['iid'] = test_ids
    agg_gtypes.set_index('iid', inplace=True, drop=False)
    agg_gtypes.loc[test_ids, 'label'] = test_df[label].values

    snps = []
    beta_df = pd.DataFrame(columns=['SNP', beta_header])
    beta_df.set_index('SNP', drop=False, inplace=True)

    for i in range(len(gene_dict['names'])):
        g = gene_dict['names'][i]
        c = gene_dict['chrom'][i]

        data = pd.read_csv('{}/{}_chr{}_{}_{}bp.csv'.format(
            sys_params['DATA_BASE_FOLDER'], label, c, g, bp), 
            dtype={'iid':int})
        data.set_index('iid', inplace=True)
        
        # Find SNP with min P value in summary statistic
        gs = [s for s in data.columns if 'rs' in s]
        gs = [s.split('_')[1] for s in gs]
        gene_betas = summary_df.loc[summary_df.index.isin(gs)]
        min_p_beta = np.argmin(gene_betas['P'].values)
        g_snp = gene_betas.iloc[min_p_beta]['SNP']
        snps.extend([g_snp])
        beta_df.loc[g_snp, 'SNP'] = g_snp
        beta_df.loc[g_snp, beta_header] = float(gene_betas.iloc[min_p_beta][beta_header])
        
        data = data.loc[data.index.isin(test_ids)][str(c)+'_'+g_snp]
        agg_gtypes.loc[test_ids, g_snp] = data
        
        print(i, g, agg_gtypes.shape, len(snps))

    # Remove chromosome prefix from the snp columns
    agg_gtypes = agg_gtypes.loc[:, ~agg_gtypes.columns.duplicated()]
    
    # Get beta values for SNPs
    betas = beta_df.loc[snps][beta_header].values
    vprint("Summary data has {} missing SNPs.".format(len(snps)-len(betas)))
    snps = beta_df.index.values
    
    # Get the minor and major alleles from the bim file
    summary_alleles = summary_df.loc[snps][['A1', 'A2']].values
    bim_alleles = maf_df.loc[snps][['A1', 'A2']].values
    assert bim_alleles.shape == summary_alleles.shape
    beta_mask = np.where(bim_alleles == summary_alleles, 
        np.ones(bim_alleles.shape), np.ones(bim_alleles.shape)*-1)
    beta_mask = beta_mask[:, 0]
    print("Flipped alleles: {}".format(np.count_nonzero(beta_mask == -1)))
    betas = betas*beta_mask
    
    gtype_mat = agg_gtypes[snps].values
    gtype_mat[gtype_mat == 2] = -1 
    gtype_mat[gtype_mat == 0] = 2
    gtype_mat[gtype_mat == -1] = 0

    print("NANs: {}".format(np.count_nonzero(np.isnan(gtype_mat))))
    vprint("Betas shape: ", betas.shape)
    vprint("Gtype_mat shape: ", gtype_mat.shape)
    assert len(betas) == gtype_mat.shape[1]

    maf = maf_df.loc[snps]['MAF'].values
    assert len(maf) == gtype_mat.shape[1], print(len(maf))
    print("min MAF, max MAF: {} {}".format(np.min(maf), np.max(maf)))

    # Replace nan with 2*MAF for that SNP
    maf = np.tile(maf, len(gtype_mat)).reshape(gtype_mat.shape)
    maf = maf*2
    vprint("MAF shape: ", maf.shape)
    gtype_mat = np.where(np.isnan(gtype_mat), maf, gtype_mat)
    print("NANs: {}".format(np.count_nonzero(np.isnan(gtype_mat))))
    
    # Get GRS for each individual as Sum[B(snp)*Geno(SNP)]
    w_gtype_mat = np.multiply(gtype_mat, betas)
    grs = np.sum(w_gtype_mat, axis=1)
    grs = np.asarray([float(x) for x in grs])

    return grs, agg_gtypes['label'].values

def results_LD_plots(chrom, known_hits, nn_hits, label, bp):
    
    # Load hapmap LD file for the chromosome
    hapmap_ld_df = pd.read_csv('/mnt/sde/HapMap_LD/ld_chr{}_CEU.csv'.format(chrom))
    vprint('HapMap LD shape: {}'.format(hapmap_ld_df.shape))

    # Load bim file for the chromosome
    if chrom <= 10:
        bim_base = '/mnt/sdd/UKBB_1/'
    else:
        bim_base = '/mnt/sdc/UKBB_2/'
    bim_fname = '{}/ukb_imp_chr{}_v3_cleaned_geno_hwe_maf_2.bim'.format(
        bim_base, chrom)
    bim_df = pd.read_csv(bim_fname, sep='\t', header=None)
    bim_df.columns = ['chrom', 'snp', 'cm', 'pos', 'a1', 'a2']
    vprint('BIM shape: {}'.format(bim_df.shape))

    # Load gene df and keep only the genes of interest
    gene_df = pd.read_csv('./params/genes.csv')
    gene_df.set_index('symbol', drop=False, inplace=True)
    gene_df.sort_values(['start'], inplace=True)
    known_df = gene_df.loc[known_hits]
    nn_df = gene_df.loc[nn_hits]

    starts = known_df['start'].to_list() + nn_df['start'].to_list()
    ends = known_df['end'].to_list() + nn_df['end'].to_list()
    
    # Get all positions between the start and end of all genes
    pi = []
    [pi.extend(np.arange(i-50e3, j+50e3+1)) for i, j in zip(starts, ends)]
    pi = np.asarray(pi)
    vprint('Positin intervals shape: {}'.format(pi.shape))

    # Retain only the SNPs belonging to the genes
    ld_df = hapmap_ld_df.loc[hapmap_ld_df['pos1'].isin(pi)]
    ld_df = ld_df.loc[ld_df['pos2'].isin(pi)]
    ld_df = pd.concat((ld_df, ld_df.rename(columns={'pos1':'pos2', 'pos2':'pos1'})))
    ld_df = ld_df[['pos1', 'pos2', 'dprime', 'r2']]
    for p in np.unique(ld_df['pos1'].values):
        row = {'pos1':p, 'pos2':p, 'dprime':np.nan, 'r2':np.nan}
        ld_df = ld_df.append(row, ignore_index=True)
    
    bim_df = bim_df.loc[bim_df['pos'].isin(pi)]
    vprint('Filtered HapMap LD shape: {}'.format(ld_df.shape))
    vprint('Filtered BIM shape: {}'.format(bim_df.shape))
    
    ld_df.sort_values(['pos1', 'pos2'], inplace=True)
    # ld_mat = pd.pivot_table(ld_df, index='pos1', columns='pos2', values='r2')
    ld_mat = pd.pivot_table(ld_df, index='pos1', columns='pos2', values='dprime')
    vprint('dprime shape: {}'.format(ld_mat.shape))
    pos = ld_mat.index.values

    ax = plt.subplot(111)
    ax.imshow(ld_mat.values, cmap='Reds', origin='lower')
    
    ticks = []
    tick_labels = []
    tick_colors = []
    for g, s, e in zip(known_df['symbol'].values, known_df['start'].values, known_df['end'].values):
        si = bisect.bisect(pos, s)
        ei = bisect.bisect(pos, e)
        ax.axvspan(si, ei, color='g', alpha=0.2)
        tick_pos = si+(ei-si)//2
        if tick_pos == len(pos):
            tick_pos -= 1
        ticks.append(tick_pos)
        tick_labels.append('{:.3f}  {}'.format(pos[tick_pos]/1e6, g))
        tick_colors.append('g')

    for g, s, e in zip(nn_df['symbol'].values, nn_df['start'].values, nn_df['end'].values):
        si = bisect.bisect(pos, s)
        ei = bisect.bisect(pos, e)
        ax.axvspan(si, ei, color='orange', alpha=0.2)
        tick_pos = si+(ei-si)//2
        if tick_pos == len(pos):
            tick_pos -= 1
        ticks.append(tick_pos)
        tick_labels.append('{:.3f}  {}'.format(pos[tick_pos]/1e6, g))
        tick_colors.append('orange')
    
    sort_ind = np.argsort(ticks)
    ticks = [ticks[i] for i in sort_ind]
    tick_labels = [tick_labels[i] for i in sort_ind]
    tick_colors = [tick_colors[i] for i in sort_ind]

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=2, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=2)

    for ticklabel, tickcolor in zip(ax.get_xticklabels(), tick_colors):
        ticklabel.set_color(tickcolor)
    for ticklabel, tickcolor in zip(ax.get_yticklabels(), tick_colors):
        ticklabel.set_color(tickcolor)

    plt.savefig('../LD_Figures_AD/ld_mat_chrom{}.png'.format(chrom), dpi=600)
    plt.close()

