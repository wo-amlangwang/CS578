# imports
import numpy as np
import pandas as pd
import gc
import time
import random
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

df_train = pd.read_csv("train_encoded.csv", sep = ',', header=0, index_col=0)
# test = pd.read_csv("test_encoded.csv", sep = ',', header=0, index_col=0)

train, val = train, test = train_test_split(df_train, test_size=0.1)

x_train = train.iloc[:,:-1]
y_train =train.iloc[:,-1]

x_val = val.iloc[:,:-1]
y_val = val.iloc[:,-1]

#LightGBM parameters
lgbm = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=2500,
    learning_rate=0.05,
    num_leaves=250,
    min_data_in_leaf=125,
    bagging_fraction=0.901,
    max_depth=13,
    reg_alpha=2.5,
    reg_lambda=2.5,
    min_split_gain=0.0001,
    min_child_weight=25,
    feature_fraction=0.5,
    silent=-1,
    verbose=-1,
    # n_jobs is set to -1 instead of 4 otherwise the kernell will time out
    n_jobs=-1)

lgbm.fit(x_train, y_train,
         eval_set=[(x_train, y_train), (x_val, y_val)],
         eval_metric='auc',
         verbose=100,
         early_stopping_rounds=100)

prediction = lgbm.predict(x_val)
print("acc: " + str(accuracy_score(y_val, prediction)))
