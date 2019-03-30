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

num_trees_arr = np.array([100,200,300])

for i in num_trees_arr:
    for j in range(5,16):
        num_trees = i
        depth = j
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), algorithm="SAMME.R", n_estimators=num_trees)
        clf = clf.fit(x_train, y_train)
        print("num of trees:" + str(i))
        print("depth:" + str(j))
        print(clf.score(x_train, y_train))
        print(clf.score(x_test, y_test))
