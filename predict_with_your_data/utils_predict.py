import sys 
from config import database,lib,shift_ ,normalization_ 
sys.path.insert(1, lib)
import nmrDataMod as ndm
import numpy as np
import pdb
# process free induction decay(fid) data and create a spectrum similar to Topspin's output
def preprocess_spectrum(path):
    data = ndm.nmrData(path, "TopSpin")
    shiftPoints = 70  # first 70 points are not actual data.
    data.leftShift(0, 1, shiftPoints)
    data.lineBroadening(1,2,20)
    data.fourierTransform(1,2)
    phase = data.autoPhase0(2,0,-50000,50000)
    data.phase(2,3, phase)
    return (np.absolute(data.allFid[3][0])).T

spectrum = []
sample_ids = []

from glob import glob
a = glob(database+"/*/")
for item in a:
    n = (preprocess_spectrum(item))
    spectrum.append(np.roll(n,shift_))
    sample_name = item.replace(database,"")
    sample_ids.append(sample_name.replace("/",""))
size = len(sample_ids)

spectra_test = np.divide(spectrum,normalization_)
dataset_len = size
