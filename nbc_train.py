import sklearn
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
# file_name = "randomSample_preprocessed.csv"
# df = pd.read_csv(file_name, sep=',', header= 0,index_col=0)
# n = int(int(len(df)*0.9))
# train = df.head(n)
# test = df.tail(len(df)-n)
# print(train.shape)
# print(test.shape)
#
# train.to_csv("random_train.csv", sep = ',')
# test.to_csv("random_test.csv", sep = ',')

train = pd.read_csv("random_train.csv", sep = ',', header=0, index_col=0)
test = pd.read_csv("random_test.csv", sep = ',', header=0, index_col=0)

x_train = train.iloc[:,:70]
y_train =train.iloc[:,-1]

x_test = test.iloc[:,:70]
y_test = test.iloc[:,-1]



from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0)
clf.fit(x_train, y_train)


print(clf.score(x_test, y_test))



