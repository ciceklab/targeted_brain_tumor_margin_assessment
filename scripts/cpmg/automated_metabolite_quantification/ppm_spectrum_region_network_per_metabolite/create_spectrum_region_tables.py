import pdb
import numpy as np
import pandas as pd
from metabolite_mapping import folder2spectrumregion, folder2ppm, folder2dataset

# convert keys from folder names to dataset entries
dataset2spectrumregion = {}
for f_key in folder2spectrumregion.keys():
    dataset2spectrumregion[folder2dataset[f_key]] = folder2spectrumregion[f_key]
dataset2ppmregion = {}
for f_key in folder2ppm.keys():
    dataset2ppmregion[folder2dataset[f_key]] = folder2ppm[f_key]

# create dataframe from region dictionaries
ppm_region_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dataset2ppmregion.items()]))
spectrum_region_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in dataset2spectrumregion.items()]))

writer = pd.ExcelWriter("./metabolite_regions.xlsx", engine="xlsxwriter")
spectrum_region_df.transpose().to_excel(writer, sheet_name="Metabolite Regions (Spectrum)", index=dataset2spectrumregion.keys())
ppm_region_df.transpose().to_excel(writer, sheet_name="Metabolite Regions (PPM)", index=dataset2ppmregion.keys())
writer.save()
pdb.set_trace()