import os
import sys
import time 
import pdb 
import pickle
import shap
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn import preprocessing 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV

from config import *
sys.path.insert(1,"../../../../")
from classification_utils import measure_model_performance
from feature_importance import plot_all_shap_spectrum
sys.path.insert(1,"../../")
from model_utils import create_data_variables

# cross validation loop
process_start = time.time()
for fold_id, (train_idx, vald_idx, test_idx, train_data, vald_data, test_data, all_data) in enumerate(generator(K, fold_dct, statistics, spectra, ppm_spectra, quant, class_labels)):
    print(f"Test fold {fold_id}")

    # normalize
    scaler = preprocessing.MinMaxScaler().fit(train_data["spectra"])
    train_data["X"] = scaler.transform(train_data["spectra"])
    vald_data["X"] = scaler.transform(vald_data["spectra"])
    test_data["X"] =  scaler.transform(test_data["spectra"])


    # grid search input generation
    cv_X = np.concatenate((train_data["X"], vald_data["X"]), axis=0)
    cv_Y = np.concatenate((train_data["class_labels"], vald_data["class_labels"]), axis=0)
    last_train_data_idx = train_data["class_labels"].shape[0]
    split = [(list(range(last_train_data_idx)), list(range(last_train_data_idx+1, cv_Y.shape[0])))]

    # train
    train_start = time.time()
    rf = RandomForestClassifier(class_weight="balanced", verbose=False, random_state=SEED)
    gs = GridSearchCV(rf, parameter_space, cv=split, verbose=0, refit=False, scoring="roc_auc", n_jobs=-1)
    gs.fit(cv_X, np.ravel(cv_Y))
    train_end = time.time()

    # refit with best parameters
    criterion = gs.best_params_["criterion"]
    max_depth = gs.best_params_["max_depth"]
    min_samples_leaf = gs.best_params_["min_samples_leaf"]
    min_samples_split = gs.best_params_["min_samples_split"]
    n_estimators = gs.best_params_["n_estimators"]
    model = RandomForestClassifier(class_weight="balanced", verbose=False, random_state=SEED, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    model.fit(cv_X, np.ravel(cv_Y))

    # save model 
    savepath = os.path.join(model_base_path, f"test_fold_{fold_id}")
    with open(savepath, "wb") as f:
        pickle.dump(model, f)

    # test
    test_start = time.time()
    test_pred = model.predict(test_data["X"])
    test_pred_prob = model.predict_proba(test_data["X"])[:,1]
    test_label = test_data["class_labels"]
    test_end = time.time()

    # calculate and plot shap values for tumor class
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_data["X"])
    figure = plot_all_shap_spectrum(shap_values[1], test_data["X"])
    figure.savefig(os.path.join(plot_base_path, f"test_fold_{fold_id}_tumor_shap_plot.pdf"))

    filepath = os.path.join(log_base_path, f"test_fold_{fold_id}_tumor_shap")
    np.save(filepath, shap_values[1])
    filepath = os.path.join(log_base_path, f"test_fold_{fold_id}_control_shap")
    np.save(filepath, shap_values[0])

    # measure performance and record metrics
    cm, auroc, aupr, precision, recall, f1, acc = measure_model_performance(test_pred, test_pred_prob, test_label)
    print("Class based accuracies: ")
    for i in range(2):
        print("Class % d: % f" %(i, cm[i,i]/np.sum(cm[i,:])))
    print("Confusion Matrix")
    print(cm)
    print("Accuracy: ",  acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("AUROC: ", auroc)
    print("AUPR: ", aupr)

    metrics["auroc"].append(auroc)
    metrics["aupr"].append(aupr)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1"].append(f1)
    metrics["acc"].append(acc)
    
    runtime["train"].append(train_end - train_start)
    runtime["test"].append(test_end - test_start)

process_end = time.time()
print(f"All training process took {process_end - process_start} secs.")
with open(os.path.join(log_base_path, "auroc.txt"), "w") as f:
    for auroc in metrics["auroc"]:
        f.write("%f\n" % auroc)
with open(os.path.join(log_base_path, "aupr.txt"), "w") as f:
    for aupr in metrics["aupr"]:
        f.write("%f\n" % aupr)
with open(os.path.join(log_base_path, "precision.txt"), "w") as f:
    for precision in metrics["precision"]:
        f.write("%f\n" % precision)
with open(os.path.join(log_base_path, "recall.txt"), "w") as f:
    for recall in metrics["recall"]:
        f.write("%f\n" % recall)
with open(os.path.join(log_base_path, "f1.txt"), "w") as f:
    for f1 in metrics["f1"]:
        f.write("%f\n" % f1)
with open(os.path.join(log_base_path, "accuracy.txt"), "w") as f:
    for accuracy in metrics["acc"]:
        f.write("%f\n" % accuracy)
with open(os.path.join(log_base_path, "train_time.txt"), "w") as f:
    for t in runtime["train"]:
        f.write("%f\n" % t)
with open(os.path.join(log_base_path, "test_time.txt"), "w") as f:
    for t in runtime["test"]:
        f.write("%f\n" % t)