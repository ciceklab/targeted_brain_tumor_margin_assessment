import os
import sys
import pandas as pd 
import numpy as np 
import pdb
import matplotlib.pyplot as plt

# import shap data of full spectrum random forest 
# seed: 35, test_fold: 2
base = '/home/doruk/glioma_quantification/cpmg/pathologic_classification/control_tumor/'
model_name = f"baseline/full_spectrum/seed_35/"
shap_path = os.path.join(base, "logs/"+model_name+"test_fold_2_tumor_shap.npy")
shap_values = np.load(shap_path)
abs_shap = np.absolute(shap_values)
max_abs_shap = np.max(abs_shap, axis=0)

# # plot region
# plt.plot(max_abs_shap[11613:11755])
# plt.savefig("temp.png")

# from inspecting the plot
region_start = 11613
region_end = 11755

# convert the region to ppm scale
# conversion from spectrum index (between 0 and 16313) to ppm (between -2 ppm to 12 ppm).
RAW_SPECTRUM_LENGTH = 16314
MIN_PPM = -2
MAX_PPM = 12
def find_ppm_value(idx):
    return round((14 * (idx) / RAW_SPECTRUM_LENGTH -2), 2)
region_start_ppm = find_ppm_value(region_start)
region_end_ppm = find_ppm_value(region_end)