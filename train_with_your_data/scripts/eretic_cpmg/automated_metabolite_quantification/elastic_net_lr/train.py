import pdb 
import os
import time
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import ElasticNet
sys.path.insert(1,"../../")
from model_utils import create_data_variables, absolute_percentage_error
from config import *

# cross validation loop
process_start = time.time()
for outer_idx, (train_idx, test_idx, train_data, test_data, all_data) in enumerate(generator(K, fold_dct, statistics, spectra, ppm_spectra, quant, class_labels)):

    # initialize model
    model = ElasticNet(random_state=SEED)
    # fit multivariate multiple linear regression model
    train_start = time.time()
    model.fit(train_data["spectra"], train_data["quant"])
    train_end = time.time()
    # save model
    with open(os.path.join(model_base_path, f"test_fold_{outer_idx}.pth"), "wb") as f:
        pickle.dump(model,f)
    # test model
    test_start = time.time()
    test_pred = model.predict(test_data["spectra"])
    test_end = time.time()
    test_label = test_data["quant"]

    # calculate and save metrics
    mse_ = mean_squared_error(test_label, test_pred, multioutput="raw_values")
    mae_ = mean_absolute_error(test_label, test_pred, multioutput="raw_values")
    r2_ = r2_score(test_label, test_pred, multioutput="raw_values")
    mape_ = mean_absolute_percentage_error(test_label, test_pred, multioutput="raw_values")
    absolute_percent_error_per_sample =  absolute_percentage_error(test_label, test_pred)
    for idx, metabolite in enumerate(metabolite_names):
        metrics["mse"][metabolite].append(mse_[idx])
        metrics["mae"][metabolite].append(mae_[idx])
        metrics["r2"][metabolite].append(r2_[idx])
        metrics["mape"][metabolite].append(mape_[idx])
        metrics["absolute_percentage_error"][metabolite] += absolute_percent_error_per_sample[:,idx].tolist()
    runtime["test"].append(test_end - test_start)
    runtime["train"].append(train_end - train_start)

    print(f"Test fold {outer_idx}:\tMAPE: {mape_}")

# save metrics and timings 
for metabolite_ind, metabolite in enumerate(metabolite_names):
    with open(os.path.join(log_base_path, f"{metabolite}__mape.txt"), "w") as f:
        for mape in metrics["mape"][metabolite]:
            f.write("%f\n" % mape)
for metabolite_ind, metabolite in enumerate(metabolite_names):
    with open(os.path.join(log_base_path, f"{metabolite}__r2.txt"), "w") as f:
        for r2 in metrics["r2"][metabolite]:
            f.write("%f\n" % r2)
for metabolite_ind, metabolite in enumerate(metabolite_names):
    with open(os.path.join(log_base_path, f"{metabolite}__mse.txt"), "w") as f:
        for mse in metrics["mse"][metabolite]:
            f.write("%f\n" % mse)
for metabolite_ind, metabolite in enumerate(metabolite_names):
    with open(os.path.join(log_base_path, f"{metabolite}__mae.txt"), "w") as f:
        for mae in metrics["mae"][metabolite]:
            f.write("%f\n" % mae)
for metabolite_ind, metabolite in enumerate(metabolite_names):
    with open(os.path.join(log_base_path, f"{metabolite}__absolute_percentage_error.txt"), "w") as f:
        for ape in metrics["absolute_percentage_error"][metabolite]:
            f.write("%f\n" % ape)
with open(os.path.join(log_base_path, "train_time.txt"), "w") as f:
    for time in runtime["train"]:
        f.write("%f\n" % time)
with open(os.path.join(log_base_path, "test_time.txt"), "w") as f:
    for time in runtime["test"]:
        f.write("%f\n" % time)
    

