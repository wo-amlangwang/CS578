import pickle
from lightgbm import LGBMClassifier

#obj = pickle.load(open('model.sav', 'rb'))
#clf = obj
#print('Feature names:', clf.feature_names())

#print('Feature importances:', list(clf.feature_importance()))
df_train = pd.read_csv("train_encoded.csv", sep = ',', header=0, index_col=0)
print(df_train.head(5))
