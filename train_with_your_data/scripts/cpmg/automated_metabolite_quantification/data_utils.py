import random
import os
import sys
import pdb
import numpy as np 
import pandas as pd
from collections import Counter, defaultdict
from pprint import pprint

# acetate scaling constant
spectrum_peak_unit_quantification = 4.886210426653892e-06 / (69.6/0.0029)  

# calculate distribution of class labels in a given dataset
def get_distribution(labels):
    label_distribution = Counter(labels)
    sum_labels = sum(label_distribution.values())
    return [f'{label_distribution[i] / sum_labels:.2%}' for i in range(np.max(labels) + 1)]

# stratified and grouped k fold cross validation
def stratified_group_k_fold(x, y, groups, k, seed=None):
    label_count = np.max(y) + 1
    label_counts_per_group = defaultdict(lambda: np.zeros(label_count))
    label_distribution = Counter()
    for label, group in zip(y, groups):
        label_counts_per_group[group][label] += 1
        label_distribution[label] += 1
    
    label_counts_per_fold = defaultdict(lambda: np.zeros(label_count))
    groups_per_fold = defaultdict(set)

    def eval_label_counts_per_fold(label_counts, fold):
        label_counts_per_fold[fold] += label_counts
        std_per_label = []
        for label in range(label_count):
            label_std = np.std([label_counts_per_fold[i][label] / label_distribution[label] for i in range(k)])
            std_per_label.append(label_std)
        label_counts_per_fold[fold] -= label_counts
        return np.mean(std_per_label)
    
    groups_and_label_counts = list(label_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_label_counts)

    for group, label_counts in sorted(groups_and_label_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_label_counts_per_fold(label_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        label_counts_per_fold[best_fold] += label_counts
        groups_per_fold[best_fold].add(group)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, group in enumerate(groups) if group in train_groups]
        test_indices = [i for i, group in enumerate(groups) if group in test_groups]

        yield train_indices, test_indices

# assign class labels to each sample in the input dataset as explained in the paper.
def create_class_labels(statistics, pos_class, neg_class):
    labels = statistics["Pathologic Classification"].tolist()
    return [1 if (x in pos_class) else 0 if (x in neg_class) else 2 for x in labels]

# split a given dataset to k folds using a stratified and grouped approach
def split_to_kfold(spectra, statistics, task, k=5, seed=35):
    # configuration
    if task == "control_tumor":
        pos_class = ["Agressive-GLIOMA", "Benign-GLIOMA"]
        neg_class = ["Control"]
    elif task == "benign_aggressive":
        pos_class = ["Agressive-GLIOMA"]
        neg_class = ["Benign-GLIOMA"]  
    else:
        print("No such task is available for this module!")
        quit()

    # variables
    patient_id = statistics["Patient ID"].tolist()
    label = create_class_labels(statistics, pos_class, neg_class)
    
    # generate fold indices
    fold_dict = {} 
    for fold, (train_idx, test_idx) in enumerate(stratified_group_k_fold(spectra, label, patient_id, k=k, seed=seed)):
        fold_dict[str(fold)] = test_idx

    return fold_dict, label


# convert from  a given ppm value (i.e. from -2 to 12 ppm) to spectrum scale (i.e. from 0 to 16313)
def ppm_to_idx(ppm):
    exact_idx = (ppm + 2) * 16314 / 14
    upper_idx = np.floor((ppm + 2.01) * 16314 / 14)
    lower_idx = np.ceil((ppm + 1.99) * 16314 / 14)
    return int(lower_idx), int(upper_idx)

# conversion between HRMAS NMR spectrum from [0, 16313] scale to [-2 ppm, 12 ppm] scale.
LOWER_PPM_BOUND, STEP, UPPER_PPM_BOUND = -2.0, 0.01, 12.0
def spectrum2ppm(spectra):
    ppm_spectra = np.zeros((spectra.shape[0], int((UPPER_PPM_BOUND - LOWER_PPM_BOUND + STEP) / STEP)))
    for ppm in np.arange(LOWER_PPM_BOUND/STEP, (UPPER_PPM_BOUND+STEP)/STEP, 1):
        ppm *= STEP
        lower_idx, upper_idx = ppm_to_idx(ppm)
        if lower_idx < 0:
            lower_idx = 0
        if upper_idx > 16313:
            upper_idx = 16313
        idx_range = range(lower_idx, upper_idx+1)
        ppm_spectra[:, int((ppm - LOWER_PPM_BOUND) / STEP)] = np.sum(spectra[:, idx_range], axis=1)
    return ppm_spectra


