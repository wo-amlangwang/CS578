import pandas as pd
import pickle as pl

from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("/Users/fangzhoulin/Documents/Data/train_10_precent_preprocessed.csv",
                 header=0, sep=",", index_col=0, parse_dates=True)

train = df.sample(frac=0.9, replace=False, random_state=1)

x = (train.iloc[:, :70]).as_matrix()
y = (train.iloc[:, -1]).as_matrix()
print(x)
print(y)

clf = MultinomialNB()
clf.fit(x, y)
'''
with open('/Users/fangzhoulin/Documents/Data/save/clf.pickle', 'wb') as f:
    pl.dump(clf, f)
'''
test = df.sample(frac=0.1, replace=False, random_state=1)

_x = (test.iloc[:, :70]).as_matrix()
_y = (test.iloc[:, -1]).as_matrix()

print(clf.score(_x, _y))