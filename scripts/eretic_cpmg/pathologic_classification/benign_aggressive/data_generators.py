import pdb
import pickle
import pandas as pd
import os 
import numpy as np
import sys
project_base_path = "/home/cicek/doruk/glioma_quantification/"
current_path = "eretic/quantification/scripts/"
sys.path.insert(1, os.path.join(project_base_path, current_path))
from data_utils import split_to_kfold, spectrum2ppm, spectrum_peak_unit_quantification
''' Custom data generator functions for fold generation with no patient and sample overlap'''


# Option #1.A: only valid PC and fully quantified samples (train, vald and test)
def cpmg_generator_1A(k, fold_dct, statistics, spectra, ppm_spectra, quant, class_labels):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = spectra[test_fold_idx,:]
        test_fold["quant"] = quant[test_fold_idx,:]
        test_fold["ppm_spectra"] = ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = class_labels[test_fold_idx,:]
        test_fold["stats"] = statistics.iloc[test_fold_idx,:].reset_index(drop=True)

        vald_fold_idx = fold_dct[str((cur_iter+1) % k)]
        vald_fold = {}
        vald_fold["spectra"] = spectra[vald_fold_idx,:]
        vald_fold["quant"] = quant[vald_fold_idx,:]
        vald_fold["ppm_spectra"] = ppm_spectra[vald_fold_idx,:]
        vald_fold["class_labels"] = class_labels[vald_fold_idx,:]
        vald_fold["stats"] = statistics.iloc[vald_fold_idx,:].reset_index(drop=True)

        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_indices.remove((cur_iter+1) % k)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] 
        train_fold = {}
        train_fold["spectra"] = spectra[train_fold_idx,:]
        train_fold["quant"] = quant[train_fold_idx,:]
        train_fold["ppm_spectra"] = ppm_spectra[train_fold_idx,:]
        train_fold["class_labels"] = class_labels[train_fold_idx,:]
        train_fold["stats"] = statistics.iloc[train_fold_idx,:].reset_index(drop=True)

        all_data = {}
        all_data["spectra"] = spectra
        all_data["quant"] = quant
        all_data["ppm_spectra"] = ppm_spectra
        all_data["class_labels"] = class_labels
        all_data["stats"] = statistics

        yield (train_fold_idx, vald_fold_idx, test_fold_idx, train_fold, vald_fold, test_fold, all_data)

        cur_iter += 1

# Option #1.B: only valid PC and fully quantified samples (train and test)
def cpmg_generator_1B(k, fold_dct, statistics, spectra, ppm_spectra, quant, class_labels):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = spectra[test_fold_idx,:]
        test_fold["quant"] = quant[test_fold_idx,:]
        test_fold["ppm_spectra"] = ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = class_labels[test_fold_idx,:]
        test_fold["stats"] = statistics.iloc[test_fold_idx,:].reset_index(drop=True)

        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] + fold_dct[str(train_fold_indices[3])]
        train_fold = {}
        train_fold["spectra"] = spectra[train_fold_idx,:]
        train_fold["quant"] = quant[train_fold_idx,:]
        train_fold["ppm_spectra"] = ppm_spectra[train_fold_idx,:]
        train_fold["class_labels"] = class_labels[train_fold_idx,:]
        train_fold["stats"] = statistics.iloc[train_fold_idx,:].reset_index(drop=True)

        all_data = {}
        all_data["spectra"] = spectra
        all_data["quant"] = quant
        all_data["ppm_spectra"] = ppm_spectra
        all_data["class_labels"] = class_labels
        all_data["stats"] = statistics

        yield (train_fold_idx, test_fold_idx, train_fold, test_fold, all_data)

        cur_iter += 1

# Option #2.A:  valid PC and fully quantified samples form test folds 
# but invalid samples are injected to the training dataset by hand (train, vald and test)
def cpmg_generator_2A(k, fold_dct, valid_statistics, valid_spectra, valid_ppm_spectra, valid_quant, valid_class_labels,\
    invalid_statistics, invalid_spectra, invalid_ppm_spectra, invalid_quant):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = spectra[test_fold_idx,:]
        test_fold["quant"] = quant[test_fold_idx,:]
        test_fold["ppm_spectra"] = ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = class_labels[test_fold_idx,:]
        test_fold["stats"] = statistics.iloc[test_fold_idx,:].reset_index(drop=True)

        vald_fold_idx = fold_dct[str((cur_iter+1) % k)]
        vald_fold = {}
        vald_fold["spectra"] = spectra[vald_fold_idx,:]
        vald_fold["quant"] = quant[vald_fold_idx,:]
        vald_fold["ppm_spectra"] = ppm_spectra[vald_fold_idx,:]
        vald_fold["class_labels"] = class_labels[vald_fold_idx,:]
        vald_fold["stats"] = statistics.iloc[vald_fold_idx,:].reset_index(drop=True)

        invalid_sample_cnt = invalid_spectra.shape[0]
        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_indices.remove((cur_iter+1) % k)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] 
        train_fold = {}
        train_fold["spectra"] = np.concat((spectra[train_fold_idx,:], invalid_spectra[:,:]), axis=0)
        train_fold["quant"] = np.concat((quant[train_fold_idx,:], invalid_quant[:,:]), axis=0)
        train_fold["ppm_spectra"] = np.concat((ppm_spectra[train_fold_idx,:], invalid_ppm_spectra[:,:]), axis=0)
        train_fold["class_labels"] = np.concat((class_labels[train_fold_idx,:], np.array([-1]*invalid_sample_cnt).reshape((-1,1))), axis=1)
        train_fold["stats"] = pd.concat([statistics.iloc[train_fold_idx,:], invalid_statistics]).reset_index(drop=True)

        all_data = {}
        all_data["spectra"] = spectra
        all_data["quant"] = quant
        all_data["ppm_spectra"] = ppm_spectra
        all_data["class_labels"] = class_labels
        all_data["stats"] = statistics

        yield (train_fold_idx, vald_fold_idx, test_fold_idx, train_fold, vald_fold, test_fold, all_data)

        cur_iter += 1

# Option #2.B:  valid PC and fully quantified samples form test folds 
# but invalid samples are injected to the training dataset by hand (train, vald and test)
def cpmg_generator_2B(k, fold_dct, valid_statistics, valid_spectra, valid_ppm_spectra, valid_quant, valid_class_labels,\
    invalid_statistics, invalid_spectra, invalid_ppm_spectra, invalid_quant):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = spectra[test_fold_idx,:]
        test_fold["quant"] = quant[test_fold_idx,:]
        test_fold["ppm_spectra"] = ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = class_labels[test_fold_idx,:]
        test_fold["stats"] = statistics.iloc[test_fold_idx,:].reset_index(drop=True)

        invalid_sample_cnt = invalid_spectra.shape[0]
        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] + fold_dct[str(train_fold_indices[3])]
        train_fold = {}
        train_fold["spectra"] = np.concat((spectra[train_fold_idx,:], invalid_spectra[:,:]), axis=0)
        train_fold["quant"] = np.concat((quant[train_fold_idx,:], invalid_quant[:,:]), axis=0)
        train_fold["ppm_spectra"] = np.concat((ppm_spectra[train_fold_idx,:], invalid_ppm_spectra[:,:]), axis=0)
        train_fold["class_labels"] = np.concat((class_labels[train_fold_idx,:], np.array([-1]*invalid_sample_cnt).reshape((-1,1))), axis=1)
        train_fold["stats"] = pd.concat([statistics.iloc[train_fold_idx,:], invalid_statistics]).reset_index(drop=True)

        all_data = {}
        all_data["spectra"] = spectra
        all_data["quant"] = quant
        all_data["ppm_spectra"] = ppm_spectra
        all_data["class_labels"] = class_labels
        all_data["stats"] = statistics

        yield (train_fold_idx, test_fold_idx, train_fold, test_fold, all_data)

        cur_iter += 1

# Option #3.A:  only valid PC samples form test folds  (train, vald and test)
def cpmg_generator_3A(k, fold_dct, statistics, spectra, ppm_spectra, quant, quant_availability, class_labels):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = spectra[test_fold_idx,:]
        test_fold["quant"] = quant[test_fold_idx,:]
        test_fold["quant_availability"] = quant_availability[test_fold_idx,:]
        test_fold["ppm_spectra"] = ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = class_labels[test_fold_idx,:]
        test_fold["stats"] = statistics.iloc[test_fold_idx,:].reset_index(drop=True)
        
        vald_fold_idx = fold_dct[str((cur_iter+1) % k)]
        vald_fold = {}
        vald_fold["spectra"] = spectra[vald_fold_idx,:]
        vald_fold["quant"] = quant[vald_fold_idx,:]
        vald_fold["quant_availability"] = quant_availability[vald_fold_idx,:]
        vald_fold["ppm_spectra"] = ppm_spectra[vald_fold_idx,:]
        vald_fold["class_labels"] = class_labels[vald_fold_idx,:]
        vald_fold["stats"] = statistics.iloc[vald_fold_idx,:].reset_index(drop=True)

        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_indices.remove((cur_iter+1) % k)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] 
        train_fold = {}
        train_fold["spectra"] = spectra[train_fold_idx,:]
        train_fold["quant"] = quant[train_fold_idx,:]
        train_fold["quant_availability"] = quant_availability[train_fold_idx,:]
        train_fold["ppm_spectra"] = ppm_spectra[train_fold_idx,:]
        train_fold["class_labels"] = class_labels[train_fold_idx,:]
        train_fold["stats"] = statistics.iloc[train_fold_idx,:].reset_index(drop=True)

        all_data = {}
        all_data["spectra"] = spectra
        all_data["quant"] = quant
        all_data["quant_availability"] = quant_availability
        all_data["ppm_spectra"] = ppm_spectra
        all_data["class_labels"] = class_labels
        all_data["stats"] = statistics

        yield (train_fold_idx, vald_fold_idx, test_fold_idx, train_fold, vald_fold, test_fold, all_data)

        cur_iter += 1

# Option #3.B:  only valid PC samples form test folds  (train and test)
def cpmg_generator_3B(k, fold_dct, statistics, spectra, ppm_spectra, quant, quant_availability, class_labels):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = spectra[test_fold_idx,:]
        test_fold["quant"] = quant[test_fold_idx,:]
        test_fold["quant_availability"] = quant_availability[test_fold_idx,:]
        test_fold["ppm_spectra"] = ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = class_labels[test_fold_idx,:]
        test_fold["stats"] = statistics.iloc[test_fold_idx,:].reset_index(drop=True)

        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] + fold_dct[str(train_fold_indices[3])]
        train_fold = {}
        train_fold["spectra"] = spectra[train_fold_idx,:]
        train_fold["quant"] = quant[train_fold_idx,:]
        train_fold["quant_availability"] = quant_availability[train_fold_idx,:]
        train_fold["ppm_spectra"] = ppm_spectra[train_fold_idx,:]
        train_fold["class_labels"] = class_labels[train_fold_idx,:]
        train_fold["stats"] = statistics.iloc[train_fold_idx,:].reset_index(drop=True)

        all_data = {}
        all_data["spectra"] = spectra
        all_data["quant"] = quant
        all_data["quant_availability"] = quant_availability
        all_data["ppm_spectra"] = ppm_spectra
        all_data["class_labels"] = class_labels
        all_data["stats"] = statistics

        yield (train_fold_idx, test_fold_idx, train_fold, test_fold, all_data)

        cur_iter += 1

# Option #4.A:  only valid PC samples form test folds 
# but invalid pc samples are injected to the training dataset
def cpmg_generator_4A(k, fold_dct, valid_statistics, valid_spectra, valid_ppm_spectra, valid_quant, valid_quant_availability, valid_class_labels,\
    invalid_statistics, invalid_spectra, invalid_ppm_spectra, invalid_quant, invalid_quant_availability):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = valid_spectra[test_fold_idx,:]
        test_fold["quant"] = valid_quant[test_fold_idx,:]
        test_fold["quant_availability"] = valid_quant_availability[test_fold_idx,:]
        test_fold["ppm_spectra"] = valid_ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = valid_class_labels[test_fold_idx,:]
        test_fold["stats"] = valid_statistics.iloc[test_fold_idx,:].reset_index(drop=True)
        
        vald_fold_idx = fold_dct[str((cur_iter+1) % k)]
        vald_fold = {}
        vald_fold["spectra"] = valid_spectra[vald_fold_idx,:]
        vald_fold["quant"] = valid_quant[vald_fold_idx,:]
        vald_fold["quant_availability"] = valid_quant_availability[vald_fold_idx,:]
        vald_fold["ppm_spectra"] = valid_ppm_spectra[vald_fold_idx,:]
        vald_fold["class_labels"] = valid_class_labels[vald_fold_idx,:]
        vald_fold["stats"] = valid_statistics.iloc[vald_fold_idx,:].reset_index(drop=True)

        invalid_sample_cnt = invalid_spectra.shape[0]
        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_indices.remove((cur_iter+1) % k)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] 
        train_fold = {}
        train_fold["spectra"] = np.concat((valid_spectra[train_fold_idx,:],invalid_spectra[:,:]), axis=1)
        train_fold["quant"] = np.concat((valid_quant[train_fold_idx,:],invalid_quant[:,:]), axis=1)
        train_fold["quant_availability"] = np.concat((valid_quant_availability[train_fold_idx,:],invalid_quant_availability[:,:]), axis=1)
        train_fold["ppm_spectra"] = np.concat((valid_ppm_spectra[train_fold_idx,:],invalid_ppm_spectra[:,:]), axis=1)
        train_fold["class_labels"] = np.concat((valid_class_labels[train_fold_idx,:],np.array([-1]*invalid_sample_cnt).reshape((-1,1))), axis=1)
        train_fold["stats"] = pd.concat([valid_statistics.iloc[train_fold_idx,:].reset_index(drop=True),invalid_statistics])

        all_data = {}
        all_data["spectra"] = valid_spectra
        all_data["quant"] = valid_quant
        all_data["quant_availability"] = valid_quant_availability
        all_data["ppm_spectra"] = valid_ppm_spectra
        all_data["class_labels"] = valid_class_labels
        all_data["stats"] = valid_statistics

        yield (train_fold_idx, vald_fold_idx, test_fold_idx, train_fold, vald_fold, test_fold, all_data)

        cur_iter += 1

# Option #4.B:  only valid PC samples form test folds 
# but invalid pc samples are injected to the training dataset
def cpmg_generator_4B(k, fold_dct, valid_statistics, valid_spectra, valid_ppm_spectra, valid_quant, valid_quant_availability, valid_class_labels,\
    invalid_statistics, invalid_spectra, invalid_ppm_spectra, invalid_quant, invalid_quant_availability):
    cur_iter = 0
    while cur_iter < k:

        test_fold_idx = fold_dct[str(cur_iter)]
        test_fold = {}
        test_fold["spectra"] = valid_spectra[test_fold_idx,:]
        test_fold["quant"] = valid_quant[test_fold_idx,:]
        test_fold["quant_availability"] = valid_quant_availability[test_fold_idx,:]
        test_fold["ppm_spectra"] = valid_ppm_spectra[test_fold_idx,:]
        test_fold["class_labels"] = valid_class_labels[test_fold_idx,:]
        test_fold["stats"] = valid_statistics.iloc[test_fold_idx,:].reset_index(drop=True)

        invalid_sample_cnt = invalid_spectra.shape[0]
        train_fold_indices = list(range(k))
        train_fold_indices.remove(cur_iter)
        train_fold_idx = [] + fold_dct[str(train_fold_indices[0])] + fold_dct[str(train_fold_indices[1])] + fold_dct[str(train_fold_indices[2])] + fold_dct[str(train_fold_indices[3])] 
        train_fold = {}
        train_fold["spectra"] = np.concat((valid_spectra[train_fold_idx,:],invalid_spectra[:,:]), axis=1)
        train_fold["quant"] = np.concat((valid_quant[train_fold_idx,:],invalid_quant[:,:]), axis=1)
        train_fold["quant_availability"] = np.concat((valid_quant_availability[train_fold_idx,:],invalid_quant_availability[:,:]), axis=1)
        train_fold["ppm_spectra"] = np.concat((valid_ppm_spectra[train_fold_idx,:],invalid_ppm_spectra[:,:]), axis=1)
        train_fold["class_labels"] = np.concat((valid_class_labels[train_fold_idx,:],np.array([-1]*invalid_sample_cnt).reshape((-1,1))), axis=1)
        train_fold["stats"] = pd.concat([valid_statistics.iloc[train_fold_idx,:].reset_index(drop=True),invalid_statistics])

        all_data = {}
        all_data["spectra"] = valid_spectra
        all_data["quant"] = valid_quant
        all_data["quant_availability"] = valid_quant_availability
        all_data["ppm_spectra"] = valid_ppm_spectra
        all_data["class_labels"] = valid_class_labels
        all_data["stats"] = valid_statistics

        yield (train_fold_idx, test_fold_idx, train_fold, test_fold, all_data)

        cur_iter += 1