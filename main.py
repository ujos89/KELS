import pandas as pd
import os
from utils.utils import *

# check dataset verify
root_dir = './preprocessed/merge/outer'
input_df = pd.read_csv(os.path.join(root_dir, 'input_merge.csv')).set_index('L2SID')
label_df = pd.read_csv(os.path.join(root_dir, 'label_merge.csv')).set_index('L2SID')

print(input_df.iloc[0])
print(input_df.iloc[0].isna())
print(label_df.iloc[0])
print(label_df.iloc[0].isna())
print()

print(label_df.iloc[0].index[label_df.iloc[0].isna()].tolist())

def get_year(series):
    nan_index = series.index[series.notna()].tolist()
    year = sorted(list(set([int(_[3]) for _ in nan_index])))
    year_col = ['L2Y'+str(y) for y in year]
    
    return year_col

series = label_df.iloc[0]

print(get_year(series))