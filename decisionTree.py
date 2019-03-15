import sklearn
import pandas as pd
import numpy as np
import pickle
import pydotplus
train = pd.read_csv("random_train.csv", sep = ',', header=0, index_col=0)
test = pd.read_csv("random_test.csv", sep = ',', header=0, index_col=0)

x_train = train.iloc[:,:70]
y_train =train.iloc[:,-1]

x_test = test.iloc[:,:70]
y_test = test.iloc[:,-1]

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=11)
clf = clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))

dot_data = tree.export_graphviz(clf, out_file=None)
graph2 = pydotplus.graph_from_dot_data(dot_data)
graph2.write_pdf("tree2.pdf")