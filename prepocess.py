import pandas as pd
from sklearn.model_selection import train_test_split
import random

# filename = "train.csv"
# print("reading data")
# n = sum(1 for line in open(filename)) - 1
# s = 10000
# skip = sorted(random.sample(range(1,n+1),n-s))
# df = pd.read_csv(filename, skiprows=skip)
#
# print(df.shape)
#
# df.dtypes
# df.to_csv("sample1.csv", sep=',')
# df = pd.read_csv("train_10_precent_preprocessed.csv", sep=',', header=0, index_col=0)
# df.drop(columns=['Census_InternalPrimaryDisplayResolutionHorizontal', 'Census_InternalPrimaryDisplayResolutionVertical'],inplace=True)
# print(df.shape)
# df.to_csv("data800k.csv", sep=',')
import pandas
import random

# filename = "train8m.csv"
# sampledFilename = "randomSample2.csv"
# n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
# s = 800000 #desired sample size
# skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
# df = pandas.read_csv(filename, skiprows=skip, index_col = 0)
# print(df.shape)
# df.to_csv(sampledFilename, sep = ',')
# # df = pandas.read_csv("randomSample.csv", header=0, index_col = 0)
# # df2 = pandas.read_csv("train8m.csv", header=0, index_col = 0)
# # print(df.loc[ '000024872c81cf03fa862aa8f99e0984' , :])
# # print(df2.loc[ '000024872c81cf03fa862aa8f99e0984' , :])
# def drop_by_name(df, name):
#     df.drop(name, axis=1, inplace=True)

train_name = "train8m.csv"
test_name = "test.csv"
df_train = pd.read_csv(train_name, sep=',', header= 0,index_col=0)
# df_test = pd.read_csv(test_name, sep=',', header= 0,index_col=0)

df_train = df_train.sample(frac=0.1, replace=False, random_state=777)

# selected_feature = ['Census_IsVirtualDevice', 'EngineVersion', 'Census_PrimaryDiskTotalCapacity', 'AVProductStatesIdentifier',
# 'Census_OSVersion', 'Census_OSUILocaleIdentifier',  'Census_TotalPhysicalRAM', 'OsBuildLab',
# 'AVProductsInstalled', 'Census_OSInstallTypeName', 'AppVersion', 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
# 'RtpStateBitfield', 'Census_ActivationChannel','HasDetections']
# drop_by_name(df, 'OrganizationIdentifier')
# drop_by_name(df, 'SmartScreen')
# drop_by_name(df, 'Census_IsWIMBootEnabled')
# drop_by_name(df, 'Census_ThresholdOptIn')
# drop_by_name(df, 'Census_InternalBatteryType')
# drop_by_name(df, 'Census_IsFlightingInternal')
# drop_by_name(df, 'DefaultBrowsersIdentifier')
# drop_by_name(df, 'Census_ProcessorClass')
# drop_by_name(df, 'PuaMode')
# drop_by_name(df, 'Census_InternalPrimaryDisplayResolutionHorizontal')
# drop_by_name(df, 'Census_InternalPrimaryDisplayResolutionVertical')
# df = df[selected_feature]
# df = df.dropna()
# print(df.head(5))
# print(df.shape)
#
# df.to_csv("train10_processed.csv", sep=',', header = True, index = True)

# df.sort_values(by = ['AvSigVersion'])
# df.to_csv("train1_processed_sorted.csv", sep=',', header = True, index = True)

train, val = train_test_split(df_train, test_size=0.1)
print(train.head(5))
print(val.head(5))
train.to_csv("train_raw.csv", sep = ',',header = True, index = True)
val.to_csv("val.csv", sep = ',',header = True, index = True)
