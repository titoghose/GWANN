
# coding: utf-8
import os
import gc
from typing import Union
import tqdm
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import seaborn as sns

from GWANN.models import *
from GWANN.dataset_utils import *

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    
    colors = {'auroc':'blue', 'acc':'orange', 'loss':'black'}
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

# General Training functions
def train_val_loop(model:nn.Module, data:dict, training_dict:dict, 
                   metric_dict:dict, log: str) -> dict:
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
    data = training_dict['data']
    optimiser = training_dict['optimiser']
    train_batch_size = training_dict['train_batch_size']
    infer_batch_size = training_dict['infer_batch_size']
    epochs = training_dict['epochs']
    early_stopping_thresh = training_dict['early_stopping']
    device = training_dict['device']
    
    datasets = {
        k: GWASDataset(
            torch.tensor(data[k][0]).float(), 
            torch.tensor(data[k][1]).float()) for k in ['train', 'val', 'test']
    }
    dataloaders = {
        'train': DataLoader(dataset=datasets['train'], 
                            batch_size=train_batch_size, shuffle=True),
        'val': DataLoader(dataset=datasets['val'], 
                          batch_size=infer_batch_size, shuffle=False),
        'test': DataLoader(dataset=datasets['test'], 
                          batch_size=infer_batch_size, shuffle=False)
    }
    metric_fns = {
        k1: {
                k2:v2().to(device) for k2, v2 in metric_dict.items()
            } for k1 in ['train', 'val', 'test']
    }
    metric_vals = {
        k1: {
                k2:torch.zeros(epochs) for k2 in metric_fns[k1].keys()
            } for k1 in ['train', 'val', 'test']
    }
    losses = {
        k1: torch.zeros(epochs) for k1 in ['train', 'val', 'test']
    }

    # Send model to device and initialise weights and metric tensors
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum').to(device)
    early_stopping = EarlyStopping(patience=early_stopping_thresh, 
                                   save_path=f'{log}/{model_name}.pt', 
                                   inc=False)

    for epoch in tqdm.tqdm(range(epochs), desc='Epoch'):
        # Train
        model.train()
        for sample in dataloaders['train']:
            model.zero_grad()
            
            X_batch = sample[0].to(device)
            y_batch = sample[1].to(device)
            
            logits = model.forward(X_batch)[:, 0]
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimiser.step()

            losses['train'][epoch] += loss.detach().item()

            for m, mf in metric_fns['train'].items():
                mf.update(logits, y_batch)
        
        # Validate
        model.eval()
        with torch.no_grad():
            for sample in dataloaders['val']:
                X_batch = sample[0].to(device)
                y_batch = sample[1].to(device)

                logits = model.forward(X_batch)[:, 0]
                loss = loss_fn(logits, y_batch)
                losses['val'][epoch] += loss.detach().item()

                for m, mf in metric_fns['val'].items():
                    mf.update(logits, y_batch)

        # Update metrics            
        for split, metric_dict in metric_fns.items():
            if split == 'test':
                continue
            for m, mf in metric_dict.items():
                metric_vals[split][m][epoch] = mf.compute().detach().cpu().item()
                mf.reset()
            losses[split][epoch] /= len(dataloaders[split].dataset)
        
        early_stopping(losses['val'][epoch], model, epoch)
        if early_stopping.early_stop:
            break

    # Test best model
    # Update the test metric for all epochs
    model.eval()
    with torch.no_grad():
        for sample in dataloaders['test']:
            X_batch = sample[0].to(device)
            y_batch = sample[1].to(device)

            logits = model.forward(X_batch)[:, 0]
            loss = loss_fn(logits, y_batch)
            losses['test'] += loss.detach().item()

            for m, mf in metric_fns['test'].items():
                mf.update(logits, y_batch)
        
    for m, mf in metric_fns['test'].items():
        metric_vals['test'][m][:] = mf.compute().detach().cpu().item()
        mf.reset()
    losses['test'] /= len(dataloaders['test'].dataset)


    # For plotting purposes, remove all values after the best epoch
    best_ep = early_stopping.best_epoch
    print('\n\n', model_name, ' BEST EPOCH ', best_ep, '\n\n')
    for split, metric_dict in metric_vals.items():
        for m, mvals in metric_dict.items():
            metric_vals[split][m] = mvals[:best_ep+1]
    for split in losses.keys():
        losses[split] = losses[split][:best_ep+1]
    
    
    return {'best_ep':best_ep, 
            'losses':losses,
            'metrics':metric_vals}

def start_training(data:dict, model_dict:dict, optim_dict:dict, train_dict:dict, 
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
    class_weights = optim_dict['class_weights']
    class_weights = torch.Tensor(class_weights).cpu()
    optimiser = optim_dict['optim']
    use_scheduler = optim_dict['use_scheduler']
    
    # Load training parameters
    train_batch_size = train_dict['train_batch_size']
    infer_batch_size = train_dict['infer_batch_size']
    epochs = train_dict['epochs']
    log = train_dict['log']
    early_stopping = train_dict['early_stopping']

    # Model initialisation
    torch.manual_seed(int(os.environ['TORCH_SEED']))
    model = construct_model(model_type, **model_args)
    
    for named_module in model.named_modules():
        if 'cov_branch' in named_module[0] and train_dict['freeze_covs']:
            print('Not initialising covariate branch')
            continue
        weight_init_linear(named_module[1])

    # Freeze covariate model weights
    if train_dict['freeze_covs'] and 'cov_branch' in model_args:
        print('Freezing covariate model')
        before_freezing = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of parameters before freezing: ', before_freezing)
        for named_param in model.named_parameters():
            if 'cov_branch' in named_param[0]:
                named_param[1].requires_grad = False
        after_freezing = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of parameters after freezing: ', after_freezing)
    
    # Optimiser initialisation
    optimiser = getattr(optim, optimiser)
    optimiser = optimiser(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=lr)
    if not use_scheduler:
        scheduler = None

    training_dict = {
        'model_name': model_name,
        'data': data,
        'optimiser':optimiser,
        'scheduler': scheduler,
        'train_batch_size': train_batch_size,
        'infer_batch_size': infer_batch_size,
        'epochs':epochs,
        'early_stopping':early_stopping,
        'class_weights':class_weights,
        'device': device
    }

    metric_dict = {
        'acc': BinaryAccuracy,
        'auroc': BinaryAUROC
    }

    return train_val_loop(model=model, data=data, training_dict=training_dict, 
                          metric_dict=metric_dict, log=log)

def train(data:dict, model_dict:dict, optim_dict:dict, 
          train_dict:dict, device:Union[int, str]) -> tuple:
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
    torch.cuda.set_device(device)
    res = start_training(
        data=data, model_dict=model_dict, 
        optim_dict=optim_dict, train_dict=train_dict, device=device)
    
    if train_dict['log'] is not None:
        metric_logs = f'{train_dict["log"]}/training_metrics.npz'
        metric_dict = {}
        for split, metric in res['metrics'].items():
            for mname, mvalue in metric.items():
                metric_dict[f'{mname}_{split}'] = mvalue
        for split, loss in res['losses'].items():
            metric_dict[f'loss_{split}'] = loss
        np.savez(metric_logs, **metric_dict)
    
    best_ep = res['best_ep']
    best_test_acc = float(res['metrics']['test']['acc'][-1])
    best_test_loss = float(res['losses']['test'][-1])
    best_test_auroc = float(res['metrics']['test']['auroc'][-1])
    
    gc.collect()
    torch.cuda.empty_cache()

    return {'Epoch':best_ep, 
            'Acc':best_test_acc, 
            'Loss':best_test_loss, 
            'AUROC':best_test_auroc}

