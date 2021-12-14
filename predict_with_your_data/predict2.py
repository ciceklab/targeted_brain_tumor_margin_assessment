import pdb
import os
import joblib
import pickle
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
from load_test_data_with_ids import quant_test, sample_ids

scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename) 

test_data = {}
test_data["X"] = quant_test
test_data["X"] =  scaler.transform(test_data["X"])

model = pickle.load(open("control_tumor_model","rb"))


tumor_data = {}
tumor_data["X"] = []
tumor_ids = []
test_pred = model.predict(test_data["X"])
test_pred_prob = model.predict_proba(test_data["X"])[:,1]
model2 = pickle.load(open("benign_aggressive_model","rb"))
test_pred2 = model2.predict(test_data["X"])
test_pred_prob2 = model2.predict_proba(test_data["X"])[:,1]

for sample in range(len(sample_ids)):
    if test_pred[sample] == 0:
        print(str(sample_ids[sample]) + " predicted as control with a probability of ("+ str(test_pred_prob[sample]))
    elif test_pred[sample] == 1:
        if test_pred2[sample] == 1:
            print(str(sample_ids[sample]) + " predicted as tumor with a probability of ("+ str(test_pred_prob[sample])+") and aggressive with a probability of ("+ str(test_pred_prob2[sample]) + ")")
        elif test_pred2[sample] == 0:
            print(str(sample_ids[sample]) + " predicted as tumor with a probability of ("+ str(test_pred_prob[sample])+") and benign with a probability of ("+ str(test_pred_prob2[sample]) + ")")


