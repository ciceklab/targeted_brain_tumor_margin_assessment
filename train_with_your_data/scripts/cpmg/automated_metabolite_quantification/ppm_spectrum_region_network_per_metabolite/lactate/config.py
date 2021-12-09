import os
import sys
import pandas as pd 
import numpy as np 
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(1,"../")
sys.path.insert(1,"../../../")
from data_generators import cpmg_generator_1A
from load_fully_quantified_cpmg_data import fq_v_ppm_spectra, fq_v_spectra, fq_v_statistics, fq_v_quant, fq_v_class_labels, fq_v_metabolite_names, fq_v_fold_dct, SEED
from metabolite_mapping import dataset2folder, folder2dataset, folder2ppmregion

# task configuration
task = "lactate"
dataset_task = folder2dataset[task]
task_target_idx = fq_v_metabolite_names.index(dataset_task)
task_feature_idx = folder2ppmregion[task]

# data configuration
ppm_spectra = fq_v_ppm_spectra[:, task_feature_idx]
spectra = fq_v_spectra
statistics = fq_v_statistics
quant = fq_v_quant[:,task_target_idx].reshape((-1,1))
class_labels = fq_v_class_labels
metabolite_names = [fq_v_metabolite_names[task_target_idx]]
fold_dct = fq_v_fold_dct
K = 5
generator = cpmg_generator_1A

# save and log configuration
base = '/home/doruk/glioma_quantification/cpmg/quantification/'
model_name = f"baseline/ppm_region_network_per_metabolite/{task}/seed_{SEED}/"
model_base_path = os.path.join(base, "models/"+model_name)
log_base_path = os.path.join(base, "logs/"+model_name)
plot_base_path = os.path.join(base, "plots/"+model_name)

# neural network model configuration
num_epochs = 2000
weight_seed = SEED
hp_space = {
    "ETA": 10**-2.1,
    "weight_decay": 0.00001
}

# gpu/cpu device selection
gpu_id = int(input("GPU index: "))
if torch.cuda.is_available():
    device = torch.device(f"cuda:{gpu_id}")
    print(f"GPU {gpu_id} is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")

# Quantification model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.all_mutual = nn.Linear(len(task_feature_idx), 10)
        self.m1 = nn.Linear(10,1)
    def forward(self, x):
        inp = F.relu(self.all_mutual(x))
        m1 = F.relu(self.m1(inp)).squeeze()
        return m1

# weight initialization
def initialize_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# measurement metric and timing storage 
metric_names = ["mae", "mse", "mape", "r2", "absolute_percentage_error"]
metrics = {}
for name in metric_names:
    metrics[name] = {}
    for metabolite_name in metabolite_names:
        metrics[name][metabolite_name] = []
timing_mode = ["train", "test"]
runtime = {}
for mode in timing_mode:
    runtime[mode] = []