
# coding: utf-8
import os
import gc
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
import seaborn as sns

from GWANN.models import *
from GWANN.dataset_utils import *

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import random
# random.seed(0)

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load paramaters
class EarlyStopping: 
    def __init__(self, patience=10, verbose=False, delta=0, save_path='checkpoint.pt', 
                 inc=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path
        self.inc = 1 if inc else -1

    def __call__(self, metric, model, epoch):
        
        score = metric*self.inc
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.__save_checkpoint__(model)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
        
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.__save_checkpoint__(model)
            self.counter = 0

    def __save_checkpoint__(self, model):
        torch.save(model, self.save_path)

class GWASDataset(Dataset):
    def __init__(self, data, labels):
        if not isinstance(data, torch.Tensor):
            self.data = torch.tensor(data, dtype=torch.float)
            self.labels = torch.tensor(labels, dtype=torch.float)
        else:
            self.data = data
            self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return data (seq_len, batch, input_dim), label for index 
        return (self.data[idx], self.labels[idx])
    
class GroupSampler(Sampler):
    def __init__(self, data_source, grp_size, random_seed):
        self.data_source = data_source
        self.grp_size = grp_size
        self.data_size = grp_size*len(self)
        random.seed(random_seed)

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        for i in range(0, self.data_size, self.grp_size):
            yield indices[i:i+self.grp_size]

    def __len__(self):
        return len(self.data_source)//self.grp_size

class BalancedBatchGroupSampler(Sampler):
    def __init__(self, dataset, batch_size, grp_size, random_seed):
        self.case_idxs = torch.where(dataset.labels==1)[0]
        self.cont_idxs = torch.where(dataset.labels==0)[0]
        
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        
        self.case_sampler = GroupSampler(self.case_idxs, 
                                         grp_size=grp_size,
                                         random_seed=random_seed)
        self.cont_sampler = GroupSampler(self.cont_idxs, 
                                         grp_size=grp_size,
                                         random_seed=random_seed)
        
    def __iter__(self):
        batch = []
        for case, cont in it.zip_longest(it.cycle(self.case_sampler), self.cont_sampler):
            if case is None or cont is None:
                break
            batch.append(self.case_idxs[case])
            batch.append(self.cont_idxs[cont])
            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch
                batch = []
            
        if len(batch) != 0:
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return int(math.ceil((len(self.cont_sampler)*2)/self.batch_size))

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
    
    colors = {'roc_auc':'blue', 'acc':'orange', 'loss':'black', 
              'snp_grads':'green'}
    tm = np.load('{}/training_metrics.npz'.format(gene_dir))
    
    metric_df = []
    for f in tm.files:
        metric = '_'.join(f.split('_')[:-1])
        if metric != m_plot:
            continue
        split = f.split('_')[-1]
        vals = tm[f]
        epochs = np.arange(len(vals))
        temp_df = pd.DataFrame.from_dict(
                    {'Epoch': epochs, 
                     m_plot: vals,
                     'Split': [split]*len(epochs)})
        metric_df.append(temp_df)

    metric_df = pd.concat(metric_df)

    fig, ax = plt.subplots()
    sns.lineplot(data=metric_df, x='Epoch', y=m_plot, style='Split', 
                 c=colors[m_plot], ax=ax)
    ax.tick_params(axis='both', labelsize=8)
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

def set_grad(model:nn.Module, requires_grad:bool) -> None:
    """Function to set the requires_grad attribute of all parameters of 
    a model.

    Parameters
    ----------
    model : nn.Module
        Model for which the requires_grad attribute needs to be set.
    requires_grad : bool
        Value to set the requires_grad attribute to.
    """
    for param in model.parameters():
        param.requires_grad = requires_grad

# Metrics
@torch.enable_grad()
def gradient_metric(model:nn.Module, loss_fn: nn.Module, X:torch.tensor, 
                    y:torch.tensor, epsilon:float=1e-8) -> torch.tensor:
    """Function to calculate the gradient of the model output with 
    respect to the input, and then return the ratio of the sum of SNP
    gradients to the sum of covariate gradients.

    Parameters
    ----------
    model : nn.Module
        Neural network model.
    loss_fn : nn.Module
        Loss function.
    X : torch.tensor
        Input data.
    y : torch.tensor
        Labels.
    epsilon : float, optional
        Value to add to the denominator to avoid division by 0
        (default=1e-8).
        
    Returns
    -------
    torch.tensor
        Sum of SNP gradients divided by the sum of covariate gradients.
    """
    
    if not hasattr(model, 'num_snps'):
        return torch.zeros_like(y).squeeze()

    model.eval()
    X.requires_grad = True
    model.zero_grad()
    raw_out = model.forward(X)[:, 0]
    loss = loss_fn(raw_out, y)
    loss.backward()
    
    snp_grads = torch.sum(
        torch.nanmean(
            torch.abs(X.grad.detach().clone()[:, :, :model.num_snps]), 
            dim=1),
        dim=1)
    # cov_grads = torch.sum(
    #     torch.nanmean(
    #         torch.abs(X.grad.detach().clone()[:, :, model.num_snps:]), 
    #         dim=1),
    #     dim=1)

    return snp_grads #(snp_grads+epsilon)/(cov_grads+epsilon)

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
    # pred = torch.argmax(torch.softmax(raw_out, dim=1), dim=1)
    pred = torch.round(torch.sigmoid(raw_out)) 
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
    
    tn = torch.sum(confusion_vector == 0).item()#*class_weights[1]
    fp = torch.sum(confusion_vector == 2).item()#*class_weights[1]
    fn = torch.sum(confusion_vector == -1).item()#*class_weights[0]
    tp = torch.sum(confusion_vector == 1).item()#*class_weights[0]
    
    return (tn, fp, fn, tp)

def metrics_from_raw(y_true:torch.tensor, pred_prob:torch.tensor) -> dict: 
    
    y_true = y_true.cpu().numpy()
    pred_prob = pred_prob.cpu().numpy()
    
    roc_auc = roc_auc_score(y_true=y_true, y_score=pred_prob)
    
    return {'roc_auc':roc_auc}

def metrics_from_conf_mat(conf_mat:Union[tuple, list, np.ndarray]) -> dict: 
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
        # acc = (tp + tn)/(tn+fp+fn+tp)
        sens = tp/(fn+tp)
        spec = tn/(tn+fp)
        acc = (sens+spec)/2
    if pos_obs:
        rec = tp/pos_obs
    if pos_pred:
        prec = tp/pos_pred
    if not np.isnan(prec+rec) and (prec+rec):
        f1 = 2*((prec*rec)/(prec+rec))
    if pos_pred and neg_pred and pos_obs and neg_obs:
        mcc = ((tp*tn) - (fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return {'f1':f1, 'prec':prec, 'rec':rec, 'acc':acc, 'mcc':mcc}

def train_val_loop(model:nn.Module, X:torch.tensor, y:torch.tensor, 
                   Xt:torch.tensor, yt:torch.tensor, training_dict:dict, 
                   log: str) -> dict:
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
    dict
        best_ep, conf_mat, avg_acc, loss, roc_auc
    """
    # Get all training requirements from training_dict
    model_name = training_dict['model_name']
    train_ind = training_dict['train_ind']
    val_ind = training_dict['val_ind']
    test_ind = training_dict['test_ind']
    loss_fn = training_dict['loss_fn']
    optimiser = training_dict['optimiser']
    batch_size = training_dict['batch_size']
    epochs = training_dict['epochs']
    early_stopping_thresh = training_dict['early_stopping']
    device = training_dict['device']
    
    train_dataset = GWASDataset(X[train_ind], y[train_ind])
    val_dataset = GWASDataset(Xt[val_ind], yt[val_ind])
    test_dataset = GWASDataset(Xt[test_ind], yt[test_ind])
    
    train_sampler = BalancedBatchGroupSampler(train_dataset, 
                                              batch_size=batch_size, 
                                              grp_size=model.grp_size, 
                                              random_seed=int(os.environ['GROUP_SEED']))
    
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_sampler=train_sampler)
    
    # Send model to device and initialise weights and metric tensors
    model.to(device)
    loss_fn = loss_fn.to(device)
    agg_conf_mat = {k:torch.zeros((epochs, 4)) for k in ['train', 'val', 'test']}
    avg_acc = {k:torch.zeros(epochs) for k in ['train', 'val', 'test']}
    avg_roc_auc = {k:torch.zeros(epochs) for k in ['train', 'val', 'test']}
    avg_loss = {k:torch.zeros(epochs) for k in ['train', 'val', 'test']}
    avg_snp_grads = {k:torch.zeros(epochs) for k in ['train', 'val', 'test']}
    early_stopping = EarlyStopping(patience=early_stopping_thresh, 
                                   save_path=f'{log}/{model_name}.pt', 
                                   inc=False)

    current_lr = optimiser.state_dict()['param_groups'][0]['lr']
    best_state = model.state_dict()
    for epoch in tqdm.tqdm(range(epochs), desc='Epoch'):
        
        # Train
        model.train()
        # for _ in range(10):
        for bnum, sample in enumerate(train_dataloader):
            model.zero_grad()
            
            X_batch = sample[0].to(device)
            y_batch = sample[1][:, 0].float().to(device)

            raw_out = model.forward(X_batch)[:, 0]
            # y_pred = pred_from_raw(raw_out.detach().clone())
            loss = loss_fn(raw_out, y_batch)
            loss.backward()
            optimiser.step()

        # Infer
        for split, dataset in zip(['train', 'test', 'val'], [train_dataset, test_dataset, val_dataset]):
            X_tensor = dataset.data
            y_tensor = dataset.labels
            
            metric_dict = infer(
                X_tensor, y_tensor, model, loss_fn, device)
            agg_conf_mat[split][epoch] = torch.as_tensor(metric_dict['conf_mat'])
            avg_loss[split][epoch] = metric_dict['loss']
            avg_roc_auc[split][epoch] = metric_dict['roc_auc']
            avg_acc[split][epoch] = metrics_from_conf_mat(agg_conf_mat[split][epoch])['acc']
            avg_snp_grads[split][epoch] = metric_dict['snp_grads']
        
        early_stopping(avg_loss['val'][epoch], model, epoch)
        # early_stopping(avg_acc[epoch][1], model, epoch)
        if early_stopping.early_stop:
            best_state = model.state_dict()
            break

    print('\n\n', model_name, ' BEST EPOCH ', early_stopping.best_epoch, '\n\n')
    
    for split in ['train', 'val', 'test']:
        agg_conf_mat[split] = agg_conf_mat[split][:min(epoch+1, epochs)].detach().cpu().numpy()
        avg_loss[split] = avg_loss[split][:min(epoch+1, epochs)].detach().cpu().numpy()
        avg_acc[split] = avg_acc[split][:min(epoch+1, epochs)].detach().cpu().numpy()
        avg_roc_auc[split] = avg_roc_auc[split][:min(epoch+1, epochs)].detach().cpu().numpy()
        avg_snp_grads[split] = avg_snp_grads[split][:min(epoch+1, epochs)].detach().cpu().numpy()

    return {'best_ep':early_stopping.best_epoch, 
            'conf_mat':agg_conf_mat, 
            'acc':avg_acc, 
            'loss':avg_loss, 
            'roc_auc':avg_roc_auc,
            'snp_grads':avg_snp_grads}

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
    # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        
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
    early_stopping = train_dict['early_stopping']

    # Convert numpy data into torch tensors
    X, y = torch.tensor(X).float(), torch.tensor(y).long()
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).long()
    
    torch.manual_seed(int(os.environ['TORCH_SEED']))
    model = construct_model(model_type, **model_args)
    for named_module in model.named_modules():
        if 'cov_model' in named_module[0]:
            continue
        weight_init_linear(named_module[1])

    freeze_cov = int(os.environ['FREEZE_COV'])
    # Freeze covariate model weights
    if freeze_cov == 1 and 'cov_model' in model_args:
        for param in model.named_parameters():
            if 'cov_model' in param[0]:
                print('Freezing cov weights')
                param[1].requires_grad = False

    loss_fn, optimiser, scheduler = training_stuff(model=model, damping=damp, 
                                        class_weights=class_weights, lr=lr, 
                                        opt=optimiser)
    if not use_scheduler:
        scheduler = None

    train_ind = np.arange(X.shape[0])
    val_splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.5, 
                                          random_state=0)
    test_ind, val_ind = next(val_splitter.split(y_test, y_test))

    training_dict = {
        'model_name': model_name,
        'train_ind': train_ind,
        'val_ind': val_ind,
        'test_ind': test_ind,
        'loss_fn': loss_fn,
        'optimiser':optimiser,
        'scheduler': scheduler,
        'batch_size': batch_size,
        'epochs':epochs,
        'early_stopping':early_stopping,
        'class_weights':class_weights,
        'device': device
    }
    
    return train_val_loop(model=model, X=X, y=y, Xt=X_test, yt=y_test, 
                        training_dict=training_dict, log=log)

@torch.no_grad()
def infer(X_tensor:torch.tensor, y_tensor:torch.tensor, model:nn.Module, 
          loss_fn:nn.Module, device:Union[str, int], perms:Optional[list]=None, 
          num_snps:int=0, class_weights:list=[1,1], batch_size:int=2048) -> dict:
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
    dict
        conf_mat : tuple of float
            (tn, fp, fn, tp)
        loss : float
            Loss.
        roc_auc : float
            ROC AUC Score
    """
    # def ActivateDropoutInEval(m):
    #     for module in m.modules():
    #         if isinstance(module, nn.Dropout):
    #             module.train() 
    # model.eval()
    # ActivateDropoutInEval(model)
    
    model.eval()
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    dataset = GWASDataset(X_tensor, y_tensor)
    sampler = BalancedBatchGroupSampler(dataset=dataset, 
                                        batch_size=batch_size, 
                                        grp_size=model.grp_size,
                                        random_seed=int(os.environ['GROUP_SEED']))
    
    dataloader = DataLoader(dataset=dataset, batch_sampler=sampler)
    
    # Iterate over the dataset 10 times. In each epoch, the grouping
    # of samples will be different, so we will get an average
    # accuracy over multiple groupings.
    y_pred = torch.tensor([], device=device).float()
    pred_prob = torch.tensor([], device=device).float()
    y_true = torch.tensor([], device=device).float()
    snp_grads = torch.tensor([], device=device).float()
    loss = 0.0
    bnum = 0

    for _ in range(1 if len(dataloader) > 1 else 20):
        for sample in dataloader:
            X_batch = sample[0].to(device)
            y_batch = sample[1][:, 0].float().to(device)
            
            raw_out = model.forward(X_batch)[:, 0]
            
            y_pred = torch.cat(
                (y_pred, pred_from_raw(raw_out.detach().clone())))
            
            pred_prob = torch.cat(
                (pred_prob, torch.sigmoid(raw_out.detach().clone())))
            
            y_true = torch.cat((y_true, y_batch.detach().clone()))
            batch_loss = loss_fn(raw_out, y_batch).detach().cpu().item()
            loss = running_avg(loss, batch_loss, bnum+1)
            
            snp_grads = torch.cat((snp_grads, 
                                    gradient_metric(
                                        model=model, loss_fn=loss_fn, 
                                        X=X_batch, y=y_batch).detach().clone()))
            # snp_grads = torch.cat((snp_grads, torch.zeros_like(y_batch)))
            bnum += 1

    class_weights = compute_class_weight(class_weight='balanced', 
                                    classes=[0, 1], 
                                    y=y_tensor.cpu().numpy())
    class_weights = torch.tensor(class_weights, device=device).float()
    conf_mat = gen_conf_mat(y_true.detach().clone(), y_pred, 
                            class_weights=class_weights)
    roc_auc = metrics_from_raw(y_true=y_true, pred_prob=pred_prob)['roc_auc']
    snp_grads = torch.mean(snp_grads).cpu().item()
                                            
    return {'conf_mat':list(conf_mat), 'loss':loss, 'roc_auc':roc_auc, 
            'snp_grads':snp_grads}

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
    print(torch.cuda.device_count())
    torch.cuda.set_device(device)
    res = start_training(
        X=X, y=y, X_test=X_test, y_test=y_test, model_dict=model_dict, 
        optim_dict=optim_dict, train_dict=train_dict, device=device)
    
    if train_dict['log'] is not None:
        metric_logs = f'{train_dict["log"]}/training_metrics.npz'
        metric_dict = {}
        for k in res.keys():
            if k == 'best_ep':
                continue
            for split, v in res[k].items():
                metric_dict[f'{k}_{split}'] = v
        np.savez(metric_logs, **metric_dict)
    
    best_ep = res['best_ep']
    best_test_acc = res['acc']['test'][best_ep]
    best_test_loss = res['loss']['test'][best_ep]
    best_test_roc_auc = res['roc_auc']['test'][best_ep]
    best_snp_grads = res['snp_grads']['test'][best_ep]
    
    gc.collect()
    torch.cuda.empty_cache()

    return {'Epoch':best_ep, 
            'Acc':best_test_acc, 
            'Loss':best_test_loss, 
            'ROC_AUC':best_test_roc_auc,
            'snp_grads':best_snp_grads}

