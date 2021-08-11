import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

def get_file_structure():
    path_dataset = './dataset'
    path_data2013 = os.path.join(path_dataset, '2013data')
    
    data2013 = sorted(os.listdir(path_data2013))
    for folder in data2013:
        if not folder.startswith('KELS'):
            data2013.remove(folder)    

    kels = []
    for idx, folder in enumerate(data2013):
        csv_files = sorted(os.listdir(os.path.join(path_data2013, data2013[idx])))
        kels.append(csv_files)
    
        # # extract file names
        # file_names = []
        # for file_ in csv_files:
        #     file_names.append(file_.split(' ')[-1].split('_')[0])
        
    # kels: [[L2Y1S, ...]]
    return kels

def get_csvs():
    path_dataset = './dataset'
    path_data2013 = os.path.join(path_dataset, '2013data')
    data2013 = sorted(os.listdir(path_data2013))
    for folder in data2013:
        if not folder.startswith('KELS'):
            data2013.remove(folder)  

    kels = get_file_structure()
    csv_files = []

    for kels_idx, folder in enumerate(kels):
        csv_year = []
        
        for idx, survey in enumerate(folder):
            if survey.endswith('.csv'):
                csv_year.append(pd.read_csv(os.path.join(path_data2013, data2013[kels_idx], survey), low_memory=False, thousands=','))
            # exec("%s = pd.read_csv(os.path.join(path_data2013, data2013[kels_idx], survey), low_memory=False, thousands=',')" % (kels[kels_idx][idx]))
            # exec("csv_year.append(%s)" % (kels[kels_idx][idx]))
        
        csv_files.append(csv_year)

    return csv_files

def preprocessing_stu(dataframe):
    
    columns = list(dataframe.columns)
    columns_del = []
    columns_basicinfo = ['L2GENDER', 'L2Y1_SCHID', 'L2Y1_REG', 'L2Y1_SCHSIZE', 'L2Y1_SCHTYPE', 'L2Y1S']
    columns_categorical = ['L2Y1S35', 'L2Y1S34', 'L2Y1S33']
    columns_0to9 =['L2Y1S28', 'L2Y1S2901', 'L2Y1S2902', 'L2Y1S31']
    columns_0or1 = ['L2Y1S22', 'L2Y1S23', 'L2Y1S24', 'L2Y1S25']
    columns_1to4 = ['L2Y1S1001', 'L2Y1S1002', 'L2Y1S1003']
    # lots of missing value in question about addicting on phone
    columns_phone = ['L2Y1S3001', 'L2Y1S3002', 'L2Y1S3003', 'L2Y1S3004', 'L2Y1S3005']

    for col in columns:
        if col.endswith('O'):
            columns_del.append(col)

        if col.startswith('L2Y1S24_'):
            columns_del.append(col)

    columns_del.extend(columns_basicinfo)
    columns_del.extend(columns_categorical)
    columns_del.extend(columns_phone)

    # delete unnecessary columns
    df_drop = dataframe.drop(columns_del, axis=1)

    # delete null value
    df_drop = df_drop[df_drop != "#NULL!"]
    # print(df_drop.shape)

    # string to int (dataframe)
    df_float = df_drop.astype(float)
    # L2SID float to int
    df_float['L2SID'] = df_float['L2SID'].apply(np.int64)

    # drop nan and invalid value
    df_prepared = df_float.dropna(axis=0) 
    df_prepared = df_prepared[df_prepared >= 0]
    df_prepared = df_prepared.dropna(axis=0) 
    df_prepared = df_prepared.set_index(['L2SID'])
    
    # split dataframe to target or not 
    columns_target = ['L2Y1_K_SCORE', 'L2Y1_E_SCORE', 'L2Y1_M_SCORE', 'L2Y1_K_CS', 'L2Y1_E_CS', 'L2Y1_M_CS', 'L2Y1_K_THETA', 'L2Y1_E_THETA','L2Y1_M_THETA']
    df_target = df_prepared[columns_target]
    df_prepared = df_prepared.drop(columns=columns_target)

    # data preprocessing
    columns_prepared = list(df_prepared.columns)

    # move value(has another sclae) to fit average
    for col in columns_prepared:
        if col in columns_0to9:
            df_prepared[col] -= 1.5
        elif col in columns_1to4:
            df_prepared[col] += .5

    # standard scaling for each row (averaging students intends)
    scaler=StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_prepared.T).T, index=df_prepared.index, columns = df_prepared.columns)

    return df_scaled, df_target

def plot_2d(df_input, df_label, label="L2Y1_E_CS"):
    df = pd.merge(df_input, df_label[label], left_index=True, right_index=True, how='left')
    df = df.rename(columns={0:'col1', 1:'col2'})
    grouped = df.groupby(label)

    plt.rcParams['figure.figsize'] = [10, 8]
    fig, ax = plt.subplots()

    for name, group in grouped:
        ax.plot(group.col1, group.col2, marker='o', linestyle='', label=name, alpha=.3)

    ax.legend(fontsize=12, loc='upper left') # legend position

    plt.title('2d Scatter', fontsize=20)
    plt.show()