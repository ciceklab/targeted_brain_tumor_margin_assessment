import os
import sys
import pandas as pd 
import numpy as np 
import pdb
sys.path.insert(1,"../")
sys.path.insert(1,"../../")
sys.path.insert(1,"../../../")

from config_u import base
from data_generators import cpmg_generator_1B
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
generator = cpmg_generator_1B

# save and log configuration
model_name = f"elastic_net_lr/seed_{SEED}/"
model_base_path = os.path.join(base, "models/cpmg/automated_metabolite_quantification/"+model_name)
log_base_path = os.path.join(base, "logs/cpmg/automated_metabolite_quantification/"+model_name)
plot_base_path = os.path.join(base, "plots/cpmg/automated_metabolite_quantification/"+model_name)

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