import sys 
import os
from config_u import lib
sys.path.insert(1,lib)
import nmrDataMod as ndm
import numpy as np

# process free induction decay(fid) data and create a spectrum similar to Topspin's output
def preprocess_spectrum(path):
    data = ndm.nmrData(path, "TopSpin")
    shiftPoints = 70  # first 70 points are not actual data.
    data.leftShift(0, 1, shiftPoints)
    data.lineBroadening(1,2,10)
    data.fourierTransform(1,2)
    phase = data.autoPhase0(2,0,-50000,50000)
    data.phase(2,3, phase)
    return np.absolute(data.allFid[3][0])