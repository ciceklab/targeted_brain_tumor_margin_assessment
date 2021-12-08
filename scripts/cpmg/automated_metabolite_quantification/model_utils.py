import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.module import _addindent
from torch.utils.data import Dataset
import numpy as np 
from collections import Counter
import pdb
import matplotlib.pyplot as plt
import os

# summarize Pytorch model
def summary(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

# multi indexing of values in a dictionary combined in a result dictionary. 
def create_data_variables(dct, idx):
    result_dct = {}
    for x in dct.keys():
        result_dct[x] = dct[x][idx]
    return result_dct
        
# Quantification dataset for all data 
class HRMASDataset_Type_A(Dataset):
    def __init__(self, dct):
        super(HRMASDataset_Type_A, self).__init__()
        self.data_dct = dct
    def __len__(self):
        return self.data_dct["spectra"].shape[0]
    def __getitem__(self, idx):
        return (self.data_dct["spectra"][idx], self.data_dct["ppm_spectra"][idx], self.data_dct["quant"][idx], self.data_dct["quant_availability"][idx], self.data_dct["class_labels"][idx])

# Quantification dataset for all data 
class HRMASDataset_Type_B(Dataset):
    def __init__(self, dct):
        super(HRMASDataset_Type_B, self).__init__()
        self.data_dct = dct
    def __len__(self):
        return self.data_dct["spectra"].shape[0]
    def __getitem__(self, idx):
        return (self.data_dct["spectra"][idx], self.data_dct["ppm_spectra"][idx], self.data_dct["quant"][idx], self.data_dct["class_labels"][idx])

# changed and used from a MIT licensed repo on github
# reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# absolute percentage error per prediction and ground truth level (no aggregation applied).
def absolute_percentage_error(y_true, y_pred):
    return np.abs(y_true - y_pred) / np.abs(y_true)