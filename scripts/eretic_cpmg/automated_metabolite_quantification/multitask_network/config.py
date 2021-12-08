import os
import sys
import pandas as pd 
import numpy as np 
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1,"../")
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

# save and log configuration
base = '/home/doruk/glioma_quantification/eretic/quantification/'
model_name = f"multitask_mlp/seed_{SEED}/"
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
    def __init__(self, metabolite_count):
        super(Model, self).__init__()
        self.metabolite_count = metabolite_count
        self.hidden = [] * self.metabolite_count 
        self.all_mutual = nn.Linear(1401, 192)

        self.m1 = nn.Linear(192,1)
        self.m2 = nn.Linear(192,1)
        self.m3 = nn.Linear(192,1)
        self.m4 = nn.Linear(192,1)
        self.m5 = nn.Linear(192,1)
        self.m6 = nn.Linear(192,1)
        self.m7 = nn.Linear(192,1)
        self.m8 = nn.Linear(192,1)
        self.m9 = nn.Linear(192,1)
        self.m10 = nn.Linear(192,1)
        self.m11 = nn.Linear(192,1)
        self.m12 = nn.Linear(192,1)
        self.m13 = nn.Linear(192,1)
        self.m14 = nn.Linear(192,1)
        self.m15 = nn.Linear(192,1)
        self.m16 = nn.Linear(192,1)
        self.m17 = nn.Linear(192,1)
        self.m18 = nn.Linear(192,1)
        self.m19 = nn.Linear(192,1)
        self.m20 = nn.Linear(192,1)
        self.m21 = nn.Linear(192,1)
        self.m22 = nn.Linear(192,1)
        self.m23 = nn.Linear(192,1)
        self.m24 = nn.Linear(192,1)
        self.m25 = nn.Linear(192,1)
        self.m26 = nn.Linear(192,1)
        self.m27 = nn.Linear(192,1)
        self.m28 = nn.Linear(192,1)
        self.m29 = nn.Linear(192,1)
        self.m30 = nn.Linear(192,1)
        self.m31 = nn.Linear(192,1)
        self.m32 = nn.Linear(192,1)
        self.m33 = nn.Linear(192,1)
        self.m34 = nn.Linear(192,1)
        self.m35 = nn.Linear(192,1)
        self.m36 = nn.Linear(192,1)
        self.m37 = nn.Linear(192,1)

  
    def forward(self, x):
        # mutual branch 
        inp = F.relu(self.all_mutual(x))

        m1 = F.relu(self.m1(inp)).squeeze()
        m2 = F.relu(self.m2(inp)).squeeze()
        m3 = F.relu(self.m3(inp)).squeeze()
        m4 = F.relu(self.m4(inp)).squeeze()
        m5 = F.relu(self.m5(inp)).squeeze()
        m6 = F.relu(self.m6(inp)).squeeze()
        m7 = F.relu(self.m7(inp)).squeeze()
        m8 = F.relu(self.m8(inp)).squeeze()
        m9 = F.relu(self.m9(inp)).squeeze()
        m10 = F.relu(self.m10(inp)).squeeze()
        m11 = F.relu(self.m11(inp)).squeeze()
        m12 = F.relu(self.m12(inp)).squeeze()
        m13 = F.relu(self.m13(inp)).squeeze()
        m14 = F.relu(self.m14(inp)).squeeze()
        m15 = F.relu(self.m15(inp)).squeeze()
        m16 = F.relu(self.m16(inp)).squeeze()
        m17 = F.relu(self.m17(inp)).squeeze()
        m18 = F.relu(self.m18(inp)).squeeze()
        m19 = F.relu(self.m19(inp)).squeeze()
        m20 = F.relu(self.m20(inp)).squeeze()
        m21 = F.relu(self.m21(inp)).squeeze()
        m22 = F.relu(self.m22(inp)).squeeze()
        m23 = F.relu(self.m23(inp)).squeeze()
        m24 = F.relu(self.m24(inp)).squeeze()
        m25 = F.relu(self.m25(inp)).squeeze()
        m26 = F.relu(self.m26(inp)).squeeze()
        m27 = F.relu(self.m27(inp)).squeeze()
        m28 = F.relu(self.m28(inp)).squeeze()
        m29 = F.relu(self.m29(inp)).squeeze()
        m30 = F.relu(self.m30(inp)).squeeze()
        m31 = F.relu(self.m31(inp)).squeeze()
        m32 = F.relu(self.m32(inp)).squeeze()
        m33 = F.relu(self.m33(inp)).squeeze()
        m34 = F.relu(self.m34(inp)).squeeze()
        m35 = F.relu(self.m35(inp)).squeeze()
        m36 = F.relu(self.m36(inp)).squeeze()
        m37 = F.relu(self.m37(inp)).squeeze()
        
        return m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20,\
            m21, m22, m23, m24, m25, m26, m27, m28, m29, m30, m31, m32, m33, m34, m35, m36, m37

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