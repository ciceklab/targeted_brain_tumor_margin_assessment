import pdb
import pickle
import pandas as pd
import os 
import numpy as np
import sys
project_base_path = "/home/doruk/glioma_quantification/"
current_path = "eretic/quantification/scripts/"
sys.path.insert(1, os.path.join(project_base_path, current_path))
from data_utils import split_to_kfold, spectrum2ppm, spectrum_peak_unit_quantification

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cpu")


SEED = int(input("(CPMG) Enter Data and Weight Initialization Seed: "))

# load fully quantified samples
datapath_base = os.path.join(project_base_path, "data/eretic/fully_quantified/") 
with open(os.path.join(datapath_base, "fully_quantified_samples_spectra"), "rb") as f:
    c_spectra = pickle.load(f)
with open(os.path.join(datapath_base, "fully_quantified_samples_quantification"), "rb") as f:
    c_quantification = pickle.load(f)
with open(os.path.join(project_base_path, "data/cpmg/metabolite_names"), "rb") as f:
    metabolite_names = pickle.load(f)
c_statistics = pd.read_pickle(os.path.join(datapath_base, "fully_quantified_samples_statistics"))

# find samples with valid pathologic classification (i.e. "*")
index = c_statistics.index
condition  = c_statistics["Pathologic Classification"] != "*"
valid_sample_indices = index[condition].tolist()
valid_statistics = c_statistics.iloc[valid_sample_indices, :].reset_index(drop=True)
valid_spectra = c_spectra[valid_sample_indices, :]
valid_quant = c_quantification[valid_sample_indices, :]


valid_pathologic_class = ["Agressive-GLIOMA", "Benign-GLIOMA", "Control"]
index = valid_statistics.index
condition = valid_statistics["Pathologic Classification"].isin(valid_pathologic_class)
task_based_sample_indices = index[condition].tolist()
statistics = valid_statistics.iloc[task_based_sample_indices, :].reset_index(drop=True)
spectra = valid_spectra[task_based_sample_indices, :]
quant = valid_quant[task_based_sample_indices, :]

# split dataset to 3 folds with no patient and sample overlap 
fold_dct, class_labels = split_to_kfold(spectra, statistics, "benign_aggressive", k=5, seed=SEED)
class_labels = np.array(class_labels).reshape(-1,1)

# convert benign aggressive class labels to control tumor classes
# currently: "benign": 0, "aggressive":1, "control":2
class_labels[class_labels == 0] = 1
class_labels[class_labels == 2] = 0
# result: "benign" and "aggressive": 1 and "control":0

# scale CPMG spectra with respect to reference Acetate and sample mass
mass = np.array(statistics["Mass"].tolist()).astype(float)
mass_factor = np.repeat(mass.reshape(-1,1), spectra.shape[1], axis=1)
normalized_spectra = np.divide(spectra, mass_factor)
scaled_spectra = normalized_spectra * spectrum_peak_unit_quantification

# calculate ppm spectra
ppm_spectra = spectrum2ppm(scaled_spectra)

# predict quantification per fold
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
model_load_base_path = "/home/doruk/glioma_quantification/eretic/quantification/models/baseline/network_per_metabolite/"
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

pred_quant = np.zeros(quant.shape)
for key in fold_dct.keys():
    # load quantifier models
    quantifiers = []
    for name in metabolite_names:
        state_dct = torch.load(os.path.join(model_load_base_path, f"{dataset2folder[name]}/seed_{SEED}/test_fold_{int(key)}.pth"), map_location=device)
        quantifiers.append(Single_Metabolite_Model())
        quantifiers[-1].load_state_dict(state_dct)
        quantifiers[-1].eval()
    model =  QuantificationWrapper(quantifiers).to(device)
    # load fold data and quantify
    sample_ids = fold_dct[key]
    input = torch.from_numpy(ppm_spectra[sample_ids,:]).float().to(device)
    result = model(input).detach().cpu().numpy()
    pred_quant[sample_ids,:] = result

# rename variables to be accessed from other scripts
fq_v_ppm_spectra = ppm_spectra
fq_v_spectra = scaled_spectra
fq_v_statistics = statistics
fq_v_quant = quant
fq_v_class_labels = class_labels
fq_v_metabolite_names = metabolite_names
fq_v_fold_dct = fold_dct
fq_v_pred_quant = pred_quant