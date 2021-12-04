import os
import joblib
import pickle
from load_test_data import ppm_spectra_test, labels_test, quant_test
import time
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from classification_utils import measure_model_performance
sys.path.insert(1,"../../../../")
sys.path.insert(1,"../../")

scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename) 

test_data = {}
test_data["X"] = quant_test
test_data["X"] =  scaler.transform(test_data["X"])
test_data["class_labels"] = np.transpose(labels_test)

base = '/mnt/gunkaynar/eretic/pathologic_classification/control_tumor/'
model_name = f"pred_quant/one_fold_RF/seed_35/"
model_base_path = os.path.join(base, "models/"+model_name)
model = pickle.load(open("control_tumor_model","rb"))





test_start = time.time()
test_pred = model.predict(test_data["X"])
test_pred_prob = model.predict_proba(test_data["X"])[:,1]
test_label = test_data["class_labels"]
test_end = time.time()

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