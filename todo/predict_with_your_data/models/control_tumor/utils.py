import sys 
sys.path.insert(1, '/Users/gunkaynar/Desktop/research/targeted_brain_tumor_margin_assessment/reproduce/lib/pyNMR')
import nmrDataMod as ndm
import numpy as np
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

database = "/Users/gunkaynar/Desktop/research/targeted_brain_tumor_margin_assessment/reproduce/predict_with_your_data/models/control_tumor/FID_data"
shift_ = -1515
normalization_ = 1.6
tumor_spectrum = []
control_spectrum = []
labels = []

n = (preprocess_spectrum(database + "/GBM2/"))
tumor_spectrum.append(np.roll(n,shift_))
labels.append(1)
n = (preprocess_spectrum(database + "/GBM3/"))
tumor_spectrum.append(np.roll(n,shift_))
labels.append(1)
n = (preprocess_spectrum(database + "/GBM5/"))
tumor_spectrum.append(np.roll(n,shift_))
labels.append(1)
n = (preprocess_spectrum(database + "/GBM6/"))
tumor_spectrum.append(np.roll(n,shift_))
labels.append(1)
n = (preprocess_spectrum(database + "/N1/"))
control_spectrum.append(np.roll(n,shift_))
labels.append(0)
n = (preprocess_spectrum(database + "/N2/"))
control_spectrum.append(np.roll(n,shift_))
labels.append(0)
n = (preprocess_spectrum(database + "/N3/"))
control_spectrum.append(np.roll(n,shift_))
labels.append(0)
n = (preprocess_spectrum(database + "/N4/"))
control_spectrum.append(np.roll(n,shift_))
labels.append(0)

size = len(labels)

labels = np.reshape(labels,(1,size))
spectrums = np.append((control_spectrum),(tumor_spectrum),axis=0)
spectra_test = spectrums/normalization_
labels_test = labels
tumor_spectrum_test = np.divide(tumor_spectrum,normalization_)
dataset_len = size
