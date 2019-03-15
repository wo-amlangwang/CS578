import sklearn
import pandas as pd
import numpy as np
import pickle

train = pd.read_csv("random_train.csv", sep = ',', header=0, index_col=0)
test = pd.read_csv("random_test.csv", sep = ',', header=0, index_col=0)

x_train = train.iloc[:,:70]
y_train =train.iloc[:,-1]

x_test = test.iloc[:,:70]
y_test = test.iloc[:,-1]

seed = 7
num_trees = 30
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=seed)
clf = clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

import pickle
# save the classifier
with open('gradientTree.pkl', 'wb') as f:
    pickle.dump(clf, f)