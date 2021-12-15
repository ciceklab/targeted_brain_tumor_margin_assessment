import random 
import os
import pdb 
import pickle
import numpy as np 
import pandas as pd 
import sys
sys.path.insert(1,"../")
sys.path.insert(1,"../../")
from config_u import base
from preprocessing_utils import preprocess_spectrum
dataset_excel_path = os.path.join(base, "data_xlsx/Table_S2.xls")
dataset_spectra_path = os.path.join(base, "data/eretic")
raw_data_path = os.path.join(base, "data/raw_data_eretic/")

# read excel sheets to pandas dataframe
statistics_df = pd.read_excel(dataset_excel_path, sheet_name="ERETIC-CPMG Data (Statistics)")
quantification_df = pd.read_excel(dataset_excel_path, sheet_name="ERETIC-CPMG Data (Metabolites)")
statistics_df["Pathologic Classification"] = statistics_df["Pathologic Classification"].replace("Control ", "Control")

# read metabolite quantifications
metabolite_names = quantification_df.columns[4:]
quantification_df[metabolite_names] = quantification_df[metabolite_names].replace("*", "0")
metabolite_quantifications = quantification_df[metabolite_names].values.astype(float)
metabolite_names = metabolite_names.values.tolist()

# create data availability matrix using metabolite quantifications
def condition(value):
    if value == 0:
        return 0
    elif value >= 0:
        return 1
quantification_availability = quantification_df[metabolite_names].copy().astype("float")
for (c_name, _) in quantification_availability.iteritems():
    quantification_availability[c_name] = quantification_availability[c_name].apply(condition)
quantification_availability =  quantification_availability.values

# read and preprocess HRMAS NMR spectra
spectra_filenames = statistics_df["Spectrum Filename"]
spectra = np.array([preprocess_spectrum(f"{dataset_spectra_path}{x}/4/").astype(float) if os.path.exists(f"{dataset_spectra_path}{x}/4/") else np.array([0]*16314) for x in spectra_filenames])

# separate eretic and cpmg data 
cpmg_idx = statistics_df.index[statistics_df['Spectrum Type'] == "Eretic-CPMG"].tolist()
cpmg_statistics = statistics_df[statistics_df.index.isin(cpmg_idx)].reset_index(drop=True)
cpmg_quantifications = metabolite_quantifications[cpmg_idx]
cpmg_quantification_availability = quantification_availability[cpmg_idx]
cpmg_spectra = spectra[cpmg_idx]

# re-label CPMG TEST samples with respect to corresponding GLIOMA instances.
cpmg_patient_ids = cpmg_statistics["Patient ID"].unique().tolist()
for pid in cpmg_patient_ids:

    all_sid = cpmg_statistics.index[cpmg_statistics['Patient ID'] == pid].tolist()
    all_samples = cpmg_statistics[cpmg_statistics.index.isin(all_sid)]
    if len(all_samples) == 1:
        continue

    glioma_sid = all_samples.index[all_samples['Group'] == "GLIOMA"].tolist()
    glioma_sample = all_samples[all_samples.index.isin(glioma_sid)]
    if len(glioma_sample) == 0:
        continue
    if len(glioma_sample) ==  len(all_samples): 
        continue
    try:
        glioma_pathologic_label = glioma_sample["Pathologic Classification"].values[0]
    except KeyError:
        pdb.set_trace()

    test_sid = all_samples.index[all_samples['Group'] == "TEST"].tolist()
    test_samples = all_samples[all_samples.index.isin(test_sid)]
    if len(test_sid) > 0:
        # update TEST instances based on rule stated in PLOS Comput Biol paper
        pos_indices = list(cpmg_statistics.loc[test_sid,:].loc[cpmg_statistics.Pathology.isin(["pos"]), "Pathologic Classification"].index)
        neg_indices = list(cpmg_statistics.loc[test_sid,:].loc[cpmg_statistics.Pathology.isin(["neg"]), "Pathologic Classification"].index)
        cpmg_statistics.loc[pos_indices,"Pathologic Classification"] = glioma_pathologic_label
        cpmg_statistics.loc[neg_indices,"Pathologic Classification"] = "Control"
        # Unit test to be used on pdb
        # temp = cpmg_statistics[["Group", "Pathology", "Pathologic Classification"]]
        # temp[temp["Group"] == "TEST"]

# save cpmg data
with open(f"{raw_data_path}fully_quantified_samples_spectra", "wb") as f:
    pickle.dump(cpmg_spectra, f)
with open(f"{raw_data_path}fully_quantified_samples_quantification", "wb") as f:
    pickle.dump(cpmg_quantifications, f)
with open(f"{raw_data_path}all_samples_quantification_availability", "wb") as f:
    pickle.dump(cpmg_quantification_availability, f)
cpmg_statistics.to_pickle(f"{raw_data_path}fully_quantified_samples_statistics")

# save order of metabolites 
with open(f"{raw_data_path}all_samples_metabolite_names", "wb") as f:
    pickle.dump(metabolite_names, f)