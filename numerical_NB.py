from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np


def numerical_nb(df):
    selected_feature = ['Census_IsVirtualDevice', 'EngineVersion',
                        'Census_OSVersion', 'Census_OSUILocaleIdentifier', 'AvSigVersion',
                        'OsBuildLab','AVProductsInstalled', 'Census_OSInstallTypeName',
                        'AppVersion','RtpStateBitfield', 'Census_ActivationChannel', 'HasDetections']
    df = df.dropna()
    df = df[selected_feature]

    y = (df.iloc[:,-1]).values
    enc = OrdinalEncoder()
    x = enc.fit_transform(df.iloc[:,:-1])
    clf = MultinomialNB()
    clf.fit(x, y)
    return clf.predict_proba(x)



df = pd.read_csv('/Users/fangzhoulin/Documents/Data/train_10_selected.csv', header=0, sep=",", index_col=0, parse_dates=True)
numerical_nb(df)
