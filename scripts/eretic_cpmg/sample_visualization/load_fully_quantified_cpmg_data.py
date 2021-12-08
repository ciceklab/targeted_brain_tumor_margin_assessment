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


SEED = int(input("(ERETIC) Enter Data and Weight Initialization Seed: "))

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

# split dataset to 5 folds with no patient and sample overlap 
fold_dct, class_labels = split_to_kfold(spectra, statistics, "benign_aggressive", k=5, seed=SEED)
class_labels = np.array(class_labels).reshape(-1,1)

# scale eretic spectra with respect to reference Acetate and sample mass
mass = np.array(statistics["Mass"].tolist()).astype(float)
mass_factor = np.repeat(mass.reshape(-1,1), spectra.shape[1], axis=1)
normalized_spectra = np.divide(spectra, mass_factor)
scaled_spectra = normalized_spectra

# calculate ppm spectra
ppm_spectra = spectrum2ppm(scaled_spectra)

# rename variables to be accessed from other scripts
# currently: "benign": 0, "aggressive":1, "control":2
fq_v_ppm_spectra = ppm_spectra
fq_v_spectra = scaled_spectra
fq_v_statistics = statistics
fq_v_quant = quant
fq_v_class_labels = class_labels
fq_v_metabolite_names = metabolite_names
fq_v_fold_dct = fold_dct