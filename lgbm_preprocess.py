# imports
import numpy as np
import pandas as pd
import gc
import time
import random
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plot
import seaborn as sb

# vars
# dataFolder = '../input/'
submissionFileName = 'submission'
trainFile = 'train_raw.csv'
testFile = 'val.csv'
# used 4000000 nr of rows in stead of 8000000 because of Kernel memory issue

seed = 6001
np.random.seed(seed)
random.seed(seed)


# def displayImportances(featureImportanceDf, submissionFileName):
#     cols = featureImportanceDf[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
#                                                                                                 ascending=False).index
#     bestFeatures = featureImportanceDf.loc[featureImportanceDf.feature.isin(cols)]
#     plot.figure(figsize=(14, 14))
#     sb.barplot(x="importance", y="feature", data=bestFeatures.sort_values(by="importance", ascending=False))
#     plot.title('LightGBM Features')
#     plot.tight_layout()
#     plot.savefig(submissionFileName + '.png')


dtypes = {
    'MachineIdentifier': 'category',
    'ProductName': 'category',
    'EngineVersion': 'category',
    'AppVersion': 'category',
    'AvSigVersion': 'category',
    'IsBeta': 'int8',
    'RtpStateBitfield': 'float16',
    'IsSxsPassiveMode': 'int8',
    'DefaultBrowsersIdentifier': 'float16',
    'AVProductStatesIdentifier': 'float32',
    'AVProductsInstalled': 'float16',
    'AVProductsEnabled': 'float16',
    'HasTpm': 'int8',
    'CountryIdentifier': 'int16',
    'CityIdentifier': 'float32',
    'OrganizationIdentifier': 'float16',
    'GeoNameIdentifier': 'float16',
    'LocaleEnglishNameIdentifier': 'int8',
    'Platform': 'category',
    'Processor': 'category',
    'OsVer': 'category',
    'OsBuild': 'int16',
    'OsSuite': 'int16',
    'OsPlatformSubRelease': 'category',
    'OsBuildLab': 'category',
    'SkuEdition': 'category',
    'IsProtected': 'float16',
    'AutoSampleOptIn': 'int8',
    'PuaMode': 'category',
    'SMode': 'float16',
    'IeVerIdentifier': 'float16',
    'SmartScreen': 'category',
    'Firewall': 'float16',
    'UacLuaenable': 'float32',
    'Census_MDC2FormFactor': 'category',
    'Census_DeviceFamily': 'category',
    'Census_OEMNameIdentifier': 'float16',
    'Census_OEMModelIdentifier': 'float32',
    'Census_ProcessorCoreCount': 'float16',
    'Census_ProcessorManufacturerIdentifier': 'float16',
    'Census_ProcessorModelIdentifier': 'float16',
    'Census_ProcessorClass': 'category',
    'Census_PrimaryDiskTotalCapacity': 'float32',
    'Census_PrimaryDiskTypeName': 'category',
    'Census_SystemVolumeTotalCapacity': 'float32',
    'Census_HasOpticalDiskDrive': 'int8',
    'Census_TotalPhysicalRAM': 'float32',
    'Census_ChassisTypeName': 'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'float16',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'float16',
    'Census_InternalPrimaryDisplayResolutionVertical': 'float16',
    'Census_PowerPlatformRoleName': 'category',
    'Census_InternalBatteryType': 'category',
    'Census_InternalBatteryNumberOfCharges': 'float32',
    'Census_OSVersion': 'category',
    'Census_OSArchitecture': 'category',
    'Census_OSBranch': 'category',
    'Census_OSBuildNumber': 'int16',
    'Census_OSBuildRevision': 'int32',
    'Census_OSEdition': 'category',
    'Census_OSSkuName': 'category',
    'Census_OSInstallTypeName': 'category',
    'Census_OSInstallLanguageIdentifier': 'float16',
    'Census_OSUILocaleIdentifier': 'int16',
    'Census_OSWUAutoUpdateOptionsName': 'category',
    'Census_IsPortableOperatingSystem': 'int8',
    'Census_GenuineStateName': 'category',
    'Census_ActivationChannel': 'category',
    'Census_IsFlightingInternal': 'float16',
    'Census_IsFlightsDisabled': 'float16',
    'Census_FlightRing': 'category',
    'Census_ThresholdOptIn': 'float16',
    'Census_FirmwareManufacturerIdentifier': 'float16',
    'Census_FirmwareVersionIdentifier': 'float32',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsWIMBootEnabled': 'float16',
    'Census_IsVirtualDevice': 'float16',
    'Census_IsTouchEnabled': 'int8',
    'Census_IsPenCapable': 'int8',
    'Census_IsAlwaysOnAlwaysConnectedCapable': 'float16',
    'Wdft_IsGamer': 'float16',
    'Wdft_RegionIdentifier': 'float16',
    'HasDetections': 'int8'
}

selectedFeatures = [
    'AVProductStatesIdentifier'
    , 'AVProductsEnabled'
    , 'IsProtected'
    , 'Processor'
    , 'OsSuite'
    , 'IsProtected'
    , 'RtpStateBitfield'
    , 'AVProductsInstalled'
    , 'Wdft_IsGamer'
    , 'DefaultBrowsersIdentifier'
    , 'OsBuild'
    , 'Wdft_RegionIdentifier'
    , 'SmartScreen'
    , 'CityIdentifier'
    , 'AppVersion'
    , 'Census_IsSecureBootEnabled'
    , 'Census_PrimaryDiskTypeName'
    , 'Census_SystemVolumeTotalCapacity'
    , 'Census_HasOpticalDiskDrive'
    , 'Census_IsWIMBootEnabled'
    , 'Census_IsVirtualDevice'
    , 'Census_IsTouchEnabled'
    , 'Census_FirmwareVersionIdentifier'
    , 'GeoNameIdentifier'
    , 'IeVerIdentifier'
    , 'Census_FirmwareManufacturerIdentifier'
    , 'Census_InternalPrimaryDisplayResolutionHorizontal'
    , 'Census_InternalPrimaryDisplayResolutionVertical'
    , 'Census_OEMModelIdentifier'
    , 'Census_ProcessorModelIdentifier'
    , 'Census_OSVersion'
    , 'Census_InternalPrimaryDiagonalDisplaySizeInInches'
    , 'Census_OEMNameIdentifier'
    , 'Census_ChassisTypeName'
    , 'Census_OSInstallLanguageIdentifier'
    , 'EngineVersion'
    , 'OrganizationIdentifier'
    , 'CountryIdentifier'
    , 'Census_ActivationChannel'
    , 'Census_ProcessorCoreCount'
    , 'Census_OSWUAutoUpdateOptionsName'
    , 'Census_InternalBatteryType'
    , 'HasDetections'
]

# Load Data with selected features
trainDf = pd.read_csv(trainFile, dtype=dtypes, usecols=selectedFeatures, low_memory=True,)
# train_labels = pd.read_csv(trainFile, usecols=['HasDetections'])

testDf = pd.read_csv(testFile, dtype=dtypes, usecols=selectedFeatures, low_memory=True)
# test_labels = pd.read_csv(testFile, usecols=['HasDetections'])

print('== Dataset Shapes ==')
print('Train : ' + str(trainDf.shape))
print('Train Labels : ' + str(trainDf.head(5)))

print('Test : ' + str(testDf.shape))
print('Test Labels : ' + str(testDf.head(5)))

numberOfRows = len(trainDf.index)
print(numberOfRows)
# Append Datasets and Cleanup
df = trainDf.append(testDf).reset_index()
del trainDf, testDf
gc.collect()

# Modify SmartScreen Feature
df.loc[df.SmartScreen == 'off', 'SmartScreen'] = 'Off'
df.loc[df.SmartScreen == 'of', 'SmartScreen'] = 'Off'
df.loc[df.SmartScreen == 'OFF', 'SmartScreen'] = 'Off'
df.loc[df.SmartScreen == '00000000', 'SmartScreen'] = 'Off'
df.loc[df.SmartScreen == '0', 'SmartScreen'] = 'Off'
df.loc[df.SmartScreen == 'ON', 'SmartScreen'] = 'On'
df.loc[df.SmartScreen == 'on', 'SmartScreen'] = 'On'
df.loc[df.SmartScreen == 'Enabled', 'SmartScreen'] = 'On'
df.loc[df.SmartScreen == 'BLOCK', 'SmartScreen'] = 'Block'
df.loc[df.SmartScreen == 'requireadmin', 'SmartScreen'] = 'RequireAdmin'
df.loc[df.SmartScreen == 'requireAdmin', 'SmartScreen'] = 'RequireAdmin'
df.loc[df.SmartScreen == 'RequiredAdmin', 'SmartScreen'] = 'RequireAdmin'
df.loc[df.SmartScreen == 'Promt', 'SmartScreen'] = 'Prompt'
df.loc[df.SmartScreen == 'Promprt', 'SmartScreen'] = 'Prompt'
df.loc[df.SmartScreen == 'prompt', 'SmartScreen'] = 'Prompt'
df.loc[df.SmartScreen == 'warn', 'SmartScreen'] = 'Warn'
df.loc[df.SmartScreen == 'Deny', 'SmartScreen'] = 'Block'
df.loc[df.SmartScreen == '&#x03;', 'SmartScreen'] = 'Off'

# Count Encoding (with exceptions)
for col in [f for f in df.columns if f not in ['index', 'HasDetections', 'Census_SystemVolumeTotalCapacity']]:
    df[col] = df[col].map(df[col].value_counts())

dfDummy = pd.get_dummies(df, dummy_na=True)
print('Dummy: ' + str(dfDummy.shape))

# Cleanup
del df
gc.collect()

# Split back to train and test
train = dfDummy[:numberOfRows]
test = dfDummy[numberOfRows:]

# Cleanup
del dfDummy
gc.collect()

# Summary Shape
print('== Dataset Shapes ==')
print('Train: ' + str(train.shape))
print('Test: ' + str(test.shape))


# Summary Columns
print('== Dataset Columns ==')
features = [f for f in train.columns if f not in ['index']]
for feature in features:
    print(feature)

train.to_csv("train_encoded.csv", sep=',', header=True, index=True)
test.to_csv("val_encoded.csv", sep=',', header=True, index=True)

    # # CV Folds
# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
#
# # Create arrays and dataframes to store results
# oofPreds = np.zeros(train.shape[0])
# subPreds = np.zeros(test.shape[0])
# featureImportanceDf = pd.DataFrame()
#
# # Loop through all Folds.
# for n_fold, (trainXId, validXId) in enumerate(folds.split(train[features], labels)):
#     # Create TrainXY and ValidationXY set based on fold-indexes
#     trainX, trainY = train[features].iloc[trainXId], labels.iloc[trainXId]
#     validX, validY = train[features].iloc[validXId], labels.iloc[validXId]
#
#     print('== Fold: ' + str(n_fold))
#
#     # LightGBM parameters
#     lgbm = LGBMClassifier(
#         objective='binary',
#         boosting_type='gbdt',
#         n_estimators=2500,
#         learning_rate=0.05,
#         num_leaves=250,
#         min_data_in_leaf=125,
#         bagging_fraction=0.901,
#         max_depth=13,
#         reg_alpha=2.5,
#         reg_lambda=2.5,
#         min_split_gain=0.0001,
#         min_child_weight=25,
#         feature_fraction=0.5,
#         silent=-1,
#         verbose=-1,
#         # n_jobs is set to -1 instead of 4 otherwise the kernell will time out
#         n_jobs=-1)
#
#     lgbm.fit(trainX, trainY,
#              eval_set=[(trainX, trainY), (validX, validY)],
#              eval_metric='auc',
#              verbose=250,
#              early_stopping_rounds=100)
#
#     oofPreds[validXId] = lgbm.predict_proba(validX, num_iteration=lgbm.best_iteration_)[:, 1]
#
#     print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(validY, oofPreds[validXId])))
#
#     # cleanup
#     print('Cleanup')
#     del trainX, trainY, validX, validY
#     gc.collect()
#
#     subPreds += lgbm.predict_proba(test[features], num_iteration=lgbm.best_iteration_)[:, 1] / folds.n_splits
#
#     # Feature Importance
#     fold_importance_df = pd.DataFrame()
#     fold_importance_df["feature"] = features
#     fold_importance_df["importance"] = lgbm.feature_importances_
#     fold_importance_df["fold"] = n_fold + 1
#     featureImportanceDf = pd.concat([featureImportanceDf, fold_importance_df], axis=0)
#
#     # cleanup
#     print('Cleanup. Post-Fold')
#     del lgbm
#     gc.collect()
#
# print('Full AUC score %.6f' % roc_auc_score(labels, oofPreds))

# Feature Importance