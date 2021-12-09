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

# load fully quantified samples
datapath_base = os.path.join(project_base_path, "data/eretic/raw/") 
with open(os.path.join(datapath_base, "all_samples_spectra"), "rb") as f:
    c_spectra = pickle.load(f)
with open(os.path.join(datapath_base, "all_samples_quantification"), "rb") as f:
    c_quantification = pickle.load(f)
with open(os.path.join(datapath_base, "all_samples_quantification_availability"), "rb") as f:
    c_quantification_availability = pickle.load(f)
with open(os.path.join(project_base_path, "data/cpmg/metabolite_names"), "rb") as f:
    metabolite_names = pickle.load(f)
c_statistics = pd.read_pickle(os.path.join(datapath_base, "all_samples_statistics"))

# find samples with invalid pathologic classification (i.e. "*")
index = c_statistics.index
condition  = c_statistics["Pathologic Classification"] == "*"
invalid_pc_idx = index[condition].tolist()
statistics = c_statistics.iloc[invalid_pc_idx, :].reset_index(drop=True)
spectra = c_spectra[invalid_pc_idx, :]
quant = c_quantification[invalid_pc_idx, :]
quant_availability = c_quantification_availability[invalid_pc_idx, :]

# scale eretic spectra with respect to reference Acetate and sample mass
mass = np.array(statistics["Mass"].tolist()).astype(float)
mass_factor = np.repeat(mass.reshape(-1,1), spectra.shape[1], axis=1)
normalized_spectra = np.divide(spectra, mass_factor)
scaled_spectra = normalized_spectra * spectrum_peak_unit_quantification

# calculate ppm spectra
ppm_spectra = spectrum2ppm(scaled_spectra)

# rename variables to be accessed from other scripts
all_i_ppm_spectra = ppm_spectra
all_i_spectra = scaled_spectra
all_i_statistics = statistics
all_i_quant = quant
all_i_quant_availability = quant_availability