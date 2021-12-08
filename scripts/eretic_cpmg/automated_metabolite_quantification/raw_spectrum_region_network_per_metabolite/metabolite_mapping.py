import pdb 
import numpy as np 
import pandas as pd
import math 


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

# find spectrum regions of all metabolites
metabolite_database_path = "/home/doruk/glioma_quantification/data/metabolite_ppm_database_baseline.xlsx"
metabolite_database = pd.read_excel(metabolite_database_path).iloc[1:,:]
metabolite_peak_start = metabolite_database["peak start (ppm)"].tolist()
metabolite_peak_end = metabolite_database["peak end (ppm)"].tolist()
metabolites = metabolite_database["Metabolite"].tolist()
unique_metabolites = np.unique(np.array(metabolites)).tolist()

# mapping from ppm to feature vector
START_PPM = -2.00
END_PPM = 12.00
STEP = 0.01
def ppm2feature(ppm):
    return int((ppm - START_PPM) / STEP)

# write metabolite regions to a dictionary
folder2ppmregion = {}
for f_key in folder2dataset.keys():
    folder2ppmregion[f_key] = []

# write metabolite regions to a dictionary
folder2spectrumregion = {}
for f_key in folder2dataset.keys():
    folder2spectrumregion[f_key] = []

def ppmidx2spectrumidx(ppm):
    exact_idx = (ppm + 2) * 16314 / 14
    upper_idx = np.floor((ppm + 2.01) * 16314 / 14)
    lower_idx = np.ceil((ppm + 1.99) * 16314 / 14)
    return int(lower_idx), int(upper_idx)

spectrum_length = 16314
shift_factor = 50
for metabolite, start, end in zip(metabolites, metabolite_peak_start, metabolite_peak_end):
    if math.isnan(start) or math.isnan(end):
        print("Invalid value:", metabolite, " ", start, " ", end)
    else:
        # metabolite peak region
        s_start, s_end = ppmidx2spectrumidx(start)
        e_start, e_end = ppmidx2spectrumidx(end)
        spr_reg = list(range(s_start, e_end+1))
        ppm_start = ppm2feature(start)
        ppm_end = ppm2feature(end)
        ppm_reg = list(range(ppm_start, ppm_end+1))
        # add left and right shift regions to account for the change in chemical shift 
        shift_at_left_end = s_start - shift_factor if s_start - shift_factor >= 0 else 0
        l_shift = list(range(shift_at_left_end, s_start))
        shift_at_right_end = e_end + shift_factor if e_end + shift_factor < spectrum_length else spectrum_length -1
        r_shift = list(range(e_end+1, shift_at_right_end))
        # concatenate shift and declared regions
        spr_reg = l_shift + spr_reg + r_shift
        # record
        folder2spectrumregion[metabolite] += spr_reg
        folder2ppmregion[metabolite] += ppm_reg

if __name__=="__main__":
    pdb.set_trace()