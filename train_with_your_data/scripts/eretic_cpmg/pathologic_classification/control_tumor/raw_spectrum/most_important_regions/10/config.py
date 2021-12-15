import os
import sys
import pandas as pd 
import numpy as np 
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(1,"../")
sys.path.insert(1,"../../")
sys.path.insert(1,"../../../")
sys.path.insert(1,"../../../../")
sys.path.insert(1,"../../../../../")
sys.path.insert(1,"../../../../../../")
sys.path.insert(1,"../../../../../../../")
sys.path.insert(1,"../../../../../../../../")

from config_u import base
from data_generators import cpmg_generator_1A
from load_fully_quantified_cpmg_data import fq_v_ppm_spectra, fq_v_spectra, fq_v_statistics, fq_v_quant, fq_v_class_labels, fq_v_metabolite_names, fq_v_fold_dct, SEED


# data configuration
ppm_spectra = fq_v_ppm_spectra
spectra = fq_v_spectra
statistics = fq_v_statistics
quant = fq_v_quant
class_labels = fq_v_class_labels
metabolite_names = fq_v_metabolite_names
fold_dct = fq_v_fold_dct
K = 5
generator = cpmg_generator_1A
index_count = 10

# random forest parameter space for grid search
n_estimators = [50, 150, 300, 400]
max_depth = [10, 15, 25, 30]
min_samples_split = [5, 10, 15]
min_samples_leaf = [2, 10, 20] 
criterion = ["gini", "entropy"]
parameter_space = dict(n_estimators = n_estimators, max_depth = max_depth,  
    min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, criterion=criterion)

# save and log configuration
model_name = f"/most_important_regions/{index_count}/seed_{SEED}/"
model_base_path = os.path.join(base, "models/eretic_cpmg/pathologic_classification/control_tumor/raw_spectrum/"+model_name)
log_base_path = os.path.join(base, "logs/eretic_cpmg/pathologic_classification/control_tumor/raw_spectrum/"+model_name)
plot_base_path = os.path.join(base, "plots/eretic_cpmg/pathologic_classification/control_tumor/raw_spectrum/"+model_name)

# measurement metric, shap values and timing storage 
all_shap_values = []
metric_names = ["auroc", "aupr", "precision", "recall", "f1", "acc"]
metrics = {}
for name in metric_names:
    metrics[name] = []
timing_mode = ["train", "test"]
runtime = {}
for mode in timing_mode:
    runtime[mode] = []