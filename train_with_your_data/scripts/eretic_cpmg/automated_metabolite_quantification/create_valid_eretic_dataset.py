import pdb
import pickle
import pandas as pd
import os 
import numpy as np
import sys
import matplotlib.pyplot as plt
project_base_path = "/home/doruk/glioma_quantification/"
current_path = "eretic/quantification/scripts/"
sys.path.insert(1, os.path.join(project_base_path, current_path))
from data_utils import split_to_kfold, spectrum2ppm, spectrum_peak_unit_quantification, ppm_to_idx


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

filenames = statistics["Spectrum Filename"].tolist()
with open("./valid_eretic_sample_filenames", "wb") as f:
    pickle.dump(filenames,f)
# # drop relevant columns and save dataset
# statistics_dataset = statistics.drop(labels=["Availability", "Recidive", "Tumor Location", "Cellularity", "Necrosis", "Infiltration (Left)", "Infiltration (Right)", "Date of Death", "Date of Last Control", "Overall Survival (days)", "Metabolite Quantification Filename"], axis="columns")
# quantification_dataset = pd.DataFrame(data=quant, columns=metabolite_names)
# pdb.set_trace()
# writer = pd.ExcelWriter("./eretic_dataset.xlsx")
# statistics_dataset.to_excel(writer, sheet_name="ERETIC-CPMG Dataset Statistics", na_rep="*", header=True, index=True)
# statistics_dataset.to_excel(writer, sheet_name="ERETIC-CPMG Dataset Statistics", na_rep="*", header=True, index=True)
# writer.save()