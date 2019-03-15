from sklearn import tree
import pandas as pd
import numpy as np
from utility import extract_feature_from_tree

from inspect import getmembers

train = pd.read_csv("/Users/fangzhoulin/Downloads/train.csv",
                 header=0, sep=",", index_col=0, parse_dates=True)

x = (train.iloc[:, :70]).as_matrix()
y = (train.iloc[:, -1]).as_matrix()

test = pd.read_csv("/Users/fangzhoulin/Downloads/test.csv",
                 header=0, sep=",", index_col=0, parse_dates=True)

_x = (test.iloc[:, :70]).as_matrix()
_y = (test.iloc[:, -1]).as_matrix()

#tree.export_graphviz(clf, out_file='tree.dot')

clf = tree.DecisionTreeClassifier(max_depth=11)
clf = clf.fit(x, y)
print(extract_feature_from_tree(clf, train.columns.values))


