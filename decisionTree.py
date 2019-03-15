import sklearn
import pandas as pd
import numpy as np
import pickle
import pydotplus
from utility import extract_feature_from_tree

train = pd.read_csv("random_train.csv", sep = ',', header=0, index_col=0)
test = pd.read_csv("random_test.csv", sep = ',', header=0, index_col=0)

x_train = train.iloc[:,:70]
y_train =train.iloc[:,-1]

x_test = test.iloc[:,:70]
y_test = test.iloc[:,-1]

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=10)
clf = clf.fit(x_train, y_train)

print(clf.score(x_test, y_test))
features = extract_feature_from_tree(clf, train.columns.values)
print(features)
print(len(features))


train2 = train[features]
test2 = test[features]

x_train = train2.iloc[:,:58]
y_train =train2.iloc[:,-1]

x_test = test2.iloc[:,:58]
y_test = test2.iloc[:,-1]

from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB(alpha=1.0)
print(x_train.head(5))
print(y_train.head(5))

# nbc.fit(x_train, y_train)
# print(nbc.score(x_test, y_test))

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph2 = pydotplus.graph_from_dot_data(dot_data)
# graph2.write_pdf("tree2.pdf")