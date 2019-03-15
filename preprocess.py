import pandas as pd
import numpy as np
import sys


def drop_by_name(df, name):
    print("[+]Droping {0}".format(name))
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
    print("[+]Encoding {0}".format(name))
    df[name] = df[name].astype('category')
    df[name] = df[name].cat.codes
    df[name] = df[name].replace(-1, np.nan)


if len(sys.argv) < 2:
    print('[-]Usage : python {0} <source-file>'.format(sys.argv[0]))
    exit()

filename = sys.argv[1]
df = pd.read_csv(filename, header=0, sep=",", index_col=0, parse_dates=True)
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

for column in df:
    if column == 'HasDetections':
        continue
    encode_category(df, column)

df = df.dropna()
_index = filename.rfind('.')
new_filename = "{0}_preprocessed.csv".format(filename[:_index])
print('[+]Storing file into {0}'.format(new_filename))
df.to_csv(new_filename, sep=',')
print('[+]Done')