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
index_count = 20
device = torch.device("cpu")

# random forest parameter space for grid search
n_estimators = [50, 150, 300, 400]
max_depth = [10, 15, 25, 30]
min_samples_split = [5, 10, 15]
min_samples_leaf = [2, 10, 20] 
criterion = ["gini", "entropy"]
parameter_space = dict(n_estimators = n_estimators, max_depth = max_depth,  
    min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, criterion=criterion)

# save and log configuration
model_name = f"most_important_metabolites/{index_count}/seed_{SEED}/"
model_base_path = os.path.join(base, "models/cpmg/pathologic_classification/benign_aggressive/metabolites/manual/"+model_name)
log_base_path = os.path.join(base, "logs/cpmg/pathologic_classification/benign_aggressive/metabolites/manual/"+model_name)
plot_base_path = os.path.join(base, "plots/cpmg/pathologic_classification/benign_aggressive/metabolites/manual/"+model_name)

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

folder2dataset = {
    "2-hg": "2-hydroxyglutarate",
    "3-hb": "Hydroxybutyrate",
    "acetate": "Acetate",
    "alanine": "Alanine",
    "allocystathionine": "Allocystathionine",
    "arginine": "Arginine",
    "ascorbate": "Ascorbate",
    "aspartate": "Aspartate",
    "betaine": "Betaine",
    "choline": "Choline",
    "creatine": "Creatine",
    "ethanolamine": "Ethanolamine",
    "gaba": "GABA",
    "glutamate": "Glutamate",
    "glutamine": "Glutamine",
    "glutathionine": "GSH",
    "glycerophosphocholine": "Glycerophosphocholine",
    "glycine": "Glycine",
    "hypotaurine": "Hypotaurine",
    "isoleucine": "Isoleucine",
    "lactate": "Lactate",
    "leucine": "Leucine",
    "lysine": "Lysine",
    "methionine": "Methionine",
    "myoinositol": "Myoinositol",
    "NAA": "NAA",
    "NAL": "NAL",
    "o-acetylcholine": "O-acetylcholine",
    "ornithine": "Ornithine",
    "phosphocholine": "Phosphocholine",
    "phosphocreatine": "Phosphocreatine",
    "proline": "Proline",
    "scylloinositol": "Scylloinositol",
    "serine": "Serine",
    "taurine": "Taurine",
    "threonine": "Threonine",
    "valine": "Valine"
}

dataset2folder = {value:key for key, value in folder2dataset.items()}

# Single Metabolite Quantification model
class Single_Metabolite_Model(nn.Module):
    def __init__(self):
        super(Single_Metabolite_Model, self).__init__()
        self.all_mutual = nn.Linear(1401, 192)
        self.m1 = nn.Linear(192,1)
    def forward(self, x):
        inp = F.relu(self.all_mutual(x))
        m1 = F.relu(self.m1(inp)).squeeze()
        return m1

# Multiple Metabolite Quantification Wrapper model
model_load_base_path = base + "/models/cpmg/automated_metabolite_quantification/full_ppm_spectrum_network_per_metabolite/"
class QuantificationWrapper(nn.Module):
    def __init__(self, quantifiers):
        super(QuantificationWrapper, self).__init__()
        self.quantifiers = quantifiers
    def forward(self, x):
        q0 = self.quantifiers[0](x)
        q1 = self.quantifiers[1](x)
        q2 = self.quantifiers[2](x)
        q3 = self.quantifiers[3](x)
        q4 = self.quantifiers[4](x)
        q5 = self.quantifiers[5](x)
        q6 = self.quantifiers[6](x)
        q7 = self.quantifiers[7](x)
        q8 = self.quantifiers[8](x)
        q9 = self.quantifiers[9](x)
        q10 = self.quantifiers[10](x)
        q11 = self.quantifiers[11](x)
        q12 = self.quantifiers[12](x)
        q13 = self.quantifiers[13](x)
        q14 = self.quantifiers[14](x)
        q15 = self.quantifiers[15](x)
        q16 = self.quantifiers[16](x)
        q17 = self.quantifiers[17](x)
        q18 = self.quantifiers[18](x)
        q19 = self.quantifiers[19](x)
        q20 = self.quantifiers[20](x)
        q21 = self.quantifiers[21](x)
        q22 = self.quantifiers[22](x)
        q23 = self.quantifiers[23](x)
        q24 = self.quantifiers[24](x)
        q25 = self.quantifiers[25](x)
        q26 = self.quantifiers[26](x)
        q27 = self.quantifiers[27](x)
        q28 = self.quantifiers[28](x)
        q29 = self.quantifiers[29](x)
        q30 = self.quantifiers[30](x)
        q31 = self.quantifiers[31](x)
        q32 = self.quantifiers[32](x)
        q33 = self.quantifiers[33](x)
        q34 = self.quantifiers[34](x)
        q35 = self.quantifiers[35](x)
        q36 = self.quantifiers[36](x)

        return torch.stack([q0,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13,q14,q15,q16,q17,q18,q19,q20,q21,q22,q23,q24,q25,q26,q27,q28,q29,q30,q31,q32,q33,q34,q35,q36]).T
