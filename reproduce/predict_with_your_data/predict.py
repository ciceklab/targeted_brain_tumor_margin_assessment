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





test_start = time.time()
test_pred = model.predict(test_data["X"])
test_pred_prob = model.predict_proba(test_data["X"])[:,1]
test_end = time.time()

test_data2 = {}
test_data2["X"] = []
model2 = pickle.load(open("benign_aggressive_model","rb"))

sample_results = {}

for i in range(0,len(sample_ids)):
    if (test_pred[i] == 1):
        test_data2["X"].append(quant_test[0])
        sample_results[sample_ids[i]] = 1
    else:
        sample_results[sample_ids[i]] = 0

test_pred2 = model2.predict(test_data2["X"])
test_pred_prob2 = model2.predict_proba(test_data2["X"])[:,1]
i = 0
for item in sample_ids:
    if(sample_results[item]==1):
        if(test_pred2[i]==1):
            sample_results[item] +=1
        i+=1
        
for item in sample_results:
    if (sample_results[item]==2):
        print(item, "is predicted as aggressive tumor")
    elif (sample_results[item]==1):
        print(item,"is predicted as benign tumor")
    elif (sample_results[item]==0):
        print(item,"is predicted as control")


