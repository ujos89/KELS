import pandas as pd
import os 
import numpy as np

path_preprocessed = 'preprocessed/preprocessed/'
file_names = sorted(os.listdir(path_preprocessed))

L2Y6S_label = pd.read_csv(os.path.join(path_preprocessed, 'L2Y6S_label.csv'))
L2Y6S_label = L2Y6S_label.set_index('L2SID', drop=True)
L2Y6S_label.index = L2Y6S_label.index.astype('int')
target_index = set(L2Y6S_label.index)
# print("target_index:", len(target_index)) # 5218

dfs = []
for file_name in file_names:
    df = pd.read_csv(os.path.join(path_preprocessed, file_name))
    df = df.set_index('L2SID', drop=True)
    df = df.sort_index(axis=1)
    if 'Unnamed' in df.columns:
        df = df.drop('Unnamed', axis=1)
        
    index_common = set(df.index) & target_index
    index_diff = target_index - index_common
    df = df.loc[index_common]
    df_diff = pd.DataFrame(np.nan, index=index_diff, columns=df.columns)
    df_ = pd.concat([df, df_diff])
    df_ = df_.sort_index(axis=0)
    
    dfs.append(df_)

for file_name, df in zip(file_names, dfs):
    if file_name.startswith("L2Y1"):
        if file_name.split('.')[0].endswith('input'):
            df_y1 = df
        else:
            df = df[['L2Y1_K_CS', 'L2Y1_E_CS', 'L2Y1_M_CS']]
            df_y1 = pd.concat([df_y1, df], axis=1)
    
    elif file_name.startswith("L2Y2"):
        if file_name.split('.')[0].endswith('input'):
            df_y2 = df
        else:
            df = df[['L2Y2_K_CS', 'L2Y2_E_CS', 'L2Y2_M_CS']]
            df_y2 = pd.concat([df_y2, df], axis=1)
    
    elif file_name.startswith("L2Y3"):
        if file_name.split('.')[0].endswith('input'):
            df_y3 = df
        else:
            df = df[['L2Y3_K_CS', 'L2Y3_E_CS', 'L2Y3_M_CS']]
            df_y3 = pd.concat([df_y3, df], axis=1)
    
    elif file_name.startswith("L2Y4"):
        if file_name.split('.')[0].endswith('input'):
            df_y4 = df
        else:
            df = df[['L2Y4_K_CS', 'L2Y4_E_CS', 'L2Y4_M_CS']]
            df_y4 = pd.concat([df_y4, df], axis=1)
        
    elif file_name.startswith("L2Y5"):
        if file_name.split('.')[0].endswith('input'):
            df_y5 = df
        else:
            df = df[['L2Y5_K_CS', 'L2Y5_E_CS', 'L2Y5_M_CS']]
            df_y5 = pd.concat([df_y5, df], axis=1)

    elif file_name.startswith("L2Y6"):
        if file_name.split('.')[0].endswith('input'):
            df_y6 = df
        else:
            df_label = df
            
## save with missing value
# df_y1.to_pickle("./preprocessed/prepared/nan/L2Y1.pkl")
# df_y2.to_pickle("./preprocessed/prepared/nan/L2Y2.pkl")
# df_y3.to_pickle("./preprocessed/prepared/nan/L2Y3.pkl")
# df_y4.to_pickle("./preprocessed/prepared/nan/L2Y4.pkl")
# df_y5.to_pickle("./preprocessed/prepared/nan/L2Y5.pkl")
# df_y6.to_pickle("./preprocessed/prepared/nan/L2Y6.pkl")
# df_label.to_pickle("./preprocessed/prepared/nan/label.pkl")

## fill missing value
df_y1_fill = df_y1.fillna(df_y1.mean())
df_y2_fill = df_y2.fillna(df_y2.mean())
df_y3_fill = df_y3.fillna(df_y3.mean())
df_y4_fill = df_y4.fillna(df_y4.mean())
df_y5_fill = df_y5.fillna(df_y5.mean())
df_y6_fill = df_y6.fillna(df_y6.mean())

# df_y1.to_pickle("./preprocessed/prepared/fill/L2Y1.pkl")
# df_y2.to_pickle("./preprocessed/prepared/fill/L2Y2.pkl")
# df_y3.to_pickle("./preprocessed/prepared/fill/L2Y3.pkl")
# df_y4.to_pickle("./preprocessed/prepared/fill/L2Y4.pkl")
# df_y5.to_pickle("./preprocessed/prepared/fill/L2Y5.pkl")
# df_y6.to_pickle("./preprocessed/prepared/fill/L2Y6.pkl")
# df_label.to_pickle("./preprocessed/prepared/fill/label.pkl")