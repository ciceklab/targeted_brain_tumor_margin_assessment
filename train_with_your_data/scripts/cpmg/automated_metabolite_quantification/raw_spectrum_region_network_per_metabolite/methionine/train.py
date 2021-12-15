import pdb 
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader
sys.path.insert(1,"../")
sys.path.insert(1,"../../")
sys.path.insert(1,"../../../")
sys.path.insert(1,"../../../../")
sys.path.insert(1,"../../../../../")
from model_utils import create_data_variables, HRMASDataset_Type_B, summary, EarlyStopping, absolute_percentage_error
from plot_loss import plot_loss
from config import *

# cross validation loop
process_start = time.time()
for outer_idx, (train_idx, vald_idx, test_idx, train_data, vald_data, test_data, all_data) in enumerate(generator(K, fold_dct, statistics, spectra, ppm_spectra, quant, class_labels)):
    print(f"Test fold {outer_idx}")
    torch.cuda.manual_seed_all(weight_seed)

    # generate data variables 
    train_dataset = HRMASDataset_Type_B(train_data)
    vald_dataset = HRMASDataset_Type_B(vald_data)
    test_dataset = HRMASDataset_Type_B(test_data)
    all_dataset = HRMASDataset_Type_B(all_data)
    all_dataloader = HRMASDataset_Type_B(all_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    vald_dataloader = DataLoader(vald_dataset, batch_size=len(vald_dataset), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    all_dataloader = DataLoader(all_dataset, batch_size=len(all_dataset))

    # initialize model and constructs
    model = Model().to(device)
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp_space["ETA"], weight_decay=hp_space["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, min_lr=1e-4, verbose=True, cooldown=25, factor=0.2)
    loss_fn_1 = nn.MSELoss()
    early_stopper = EarlyStopping(patience=100)
    print(summary(model))

    # training loop
    train_losses = {}
    for idx, metabolite in enumerate(metabolite_names):
        train_losses[metabolite] = []
    vald_losses = {}
    for idx, metabolite in enumerate(metabolite_names):
        vald_losses[metabolite] = []
    train_start = time.time()
    for epoch in range(num_epochs):
        # training
        train_loss = {}
        for idx, metabolite in enumerate(metabolite_names):
            train_loss[metabolite] = 0
        for batch_idx, (_, ppm_spectra, target, _) in enumerate(train_dataloader):
            ppm_spectra = ppm_spectra.float().to(device)
            target = target.float().to(device)
            optimizer.zero_grad()
            preds = model(ppm_spectra)
            loss = None
            for idx, metabolite in enumerate(metabolite_names):
                temp = loss_fn_1(preds, target[:,idx].squeeze()) 
                train_loss[metabolite] += temp.item()
                if loss is None:
                    loss = temp
                else:
                    loss += temp
            loss.backward()
            optimizer.step()
        for idx, metabolite in enumerate(metabolite_names):
            train_loss[metabolite] /= (batch_idx+1)
            train_losses[metabolite].append(train_loss[metabolite])
        total_train_loss = 0
        for idx, metabolite in enumerate(metabolite_names):
            total_train_loss += train_loss[metabolite]
        # validation
        vald_loss = {}
        for idx, metabolite in enumerate(metabolite_names):
            vald_loss[metabolite] = 0
        for batch_idx, (_, ppm_spectra, target, _) in enumerate(vald_dataloader):
            ppm_spectra = ppm_spectra.float().to(device)
            target = target.float().to(device)
            with torch.no_grad():
                preds = model(ppm_spectra)
                loss = None
                for idx, metabolite in enumerate(metabolite_names):
                    temp = loss_fn_1(preds, target[:,idx].squeeze()) 
                    vald_loss[metabolite] += temp.item()
                    if loss is None:
                        loss = temp
                    else:
                        loss += temp
        for idx, metabolite in enumerate(metabolite_names):
            vald_loss[metabolite] /= (batch_idx+1)
            vald_losses[metabolite].append(vald_loss[metabolite])
        total_vald_loss = 0
        for idx, metabolite in enumerate(metabolite_names):
            total_vald_loss += vald_loss[metabolite]

        scheduler.step(total_vald_loss)
        if epoch % 5 == 0:
            print(f"Epoch: {epoch}\tTrain Loss (MSE): {total_train_loss}\tValidation Loss (MSE): {total_vald_loss}")
        # early stopping
        early_stopper(total_vald_loss )
        if early_stopper.early_stop == True:
            print(f"Early stopping at epoch {epoch}")
            break
    train_end = time.time()

    # save model 
    savepath = os.path.join(model_base_path, f"test_fold_{outer_idx}.pth")
    torch.save(model.state_dict(), savepath)

    # test model on test dataset 
    test_start = time.time()
    for batch_idx, (_, ppm_spectra, target, _) in enumerate(test_dataloader):
        ppm_spectra = ppm_spectra.float().to(device)
        target = target.float().to(device)
        with torch.no_grad():
            preds = model(ppm_spectra)
            test_label = target.squeeze()
    test_end = time.time()
    test_label = test_label.detach().cpu().numpy()
    test_pred = preds.detach().cpu().numpy()

    # calculate and save metrics
    mse_ = mean_squared_error(test_label, test_pred, multioutput="raw_values")
    mae_ = mean_absolute_error(test_label, test_pred, multioutput="raw_values")
    r2_ = r2_score(test_label, test_pred, multioutput="raw_values")
    mape_ = mean_absolute_percentage_error(test_label, test_pred, multioutput="raw_values")
    absolute_percent_error_per_sample =  absolute_percentage_error(test_label, test_pred)
    for idx, metabolite in enumerate(metabolite_names):
        metrics["mse"][metabolite].append(mse_)
        metrics["mae"][metabolite].append(mae_)
        metrics["r2"][metabolite].append(r2_)
        metrics["mape"][metabolite].append(mape_)
        metrics["absolute_percentage_error"][metabolite] += absolute_percent_error_per_sample[:].tolist()
    runtime["test"].append(test_end - test_start)
    runtime["train"].append(train_end - train_start)

    # plot and save loss graph
    plot_loss(train_losses, vald_losses, outer_idx, plot_base_path, metabolite_names, model_name)
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