import pandas as pd
import numpy as np


def drop_by_name(df, name):
    df.drop(name, axis=1, inplace=True)


def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    target_type = df[target].dtypes
    target_type = target_type[0] if hasattr(target_type, '__iter__') else target_type

    if target_type in (np.int64, np.int32):

        dummies = pd.get_dummies(df[target])
        return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)
    else:
        return df.as_matrix(result).astype(np.float32), df.as_matrix([target]).astype(np.float32)


def encode_category(df, name):
    df[name] = df[name].astype('category')
    df[name] = df[name].cat.codes
    df[name] = df[name].replace(-1, np.nan)
    print("encoding {0}".format(name))


df = pd.read_csv("/Users/fangzhoulin/Documents/Data/train_10_precent.csv",
                 header=0, sep=",", index_col=0, parse_dates=True)
drop_by_name(df, 'OrganizationIdentifier')
drop_by_name(df, 'SmartScreen')
drop_by_name(df, 'Census_IsWIMBootEnabled')
drop_by_name(df, 'Census_ThresholdOptIn')
drop_by_name(df, 'Census_InternalBatteryType')
drop_by_name(df, 'Census_IsFlightingInternal')
drop_by_name(df, 'DefaultBrowsersIdentifier')
drop_by_name(df, 'Census_ProcessorClass')
drop_by_name(df, 'PuaMode')
drop_by_name(df, 'Census_InternalPrimaryDisplayResolutionHorizontal')
drop_by_name(df, 'Census_InternalPrimaryDisplayResolutionVertical')

'''
encode_category(df, 'AvSigVersion')
encode_category(df, 'AppVersion')
encode_category(df, 'Census_FlightRing')
encode_category(df, 'EngineVersion')
encode_category(df, 'Census_ActivationChannel')
encode_category(df, 'Census_GenuineStateName')
encode_category(df, 'Census_OSWUAutoUpdateOptionsName')
encode_category(df, 'Census_OSSkuName')
encode_category(df, 'OsVer')
encode_category(df, 'Census_OSInstallTypeName')
encode_category(df, 'Census_OSEdition')
encode_category(df, 'OsPlatformSubRelease')
encode_category(df, 'OsBuildLab')
encode_category(df, 'Census_OSBranch')
encode_category(df, 'Census_OSArchitecture')
encode_category(df, 'Census_OSVersion')
encode_category(df, 'SkuEdition')
encode_category(df, 'Census_PowerPlatformRoleName')
encode_category(df, 'Census_ChassisTypeName')
encode_category(df, 'Census_MDC2FormFactor')
encode_category(df, 'Census_PrimaryDiskTypeName')
encode_category(df, 'Census_DeviceFamily')
encode_category(df, 'Processor')
encode_category(df, 'ProductName')
encode_category(df, 'Platform')
encode_category(df, 'Census_InternalBatteryNumberOfCharges')
'''
for column in df:
    if column == 'HasDetections':
        continue
    encode_category(df, column)

df = df.dropna()

print(df.isnull().sum().sort_values())

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head(5))
print(df.shape)
df.to_csv("/Users/fangzhoulin/Documents/Data/train_10_precent_preprocessed.csv", sep=',')
