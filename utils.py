import numpy as np
import pandas as pd
import os

def get_file_structure():
    path_dataset = './dataset'
    path_data2013 = os.path.join(path_dataset, '2013data')

    folders = os.listdir(path_dataset)
    data2013 = sorted(os.listdir(path_data2013))

    kels_raw = []
    kels = []
    for idx, folder in enumerate(data2013):
        csv_files = sorted(os.listdir(os.path.join(path_data2013, data2013[idx])))
        
        # extract file names
        file_names = []
        for file_ in csv_files:
            file_names.append(file_.split(' ')[-1].split('_')[0])
        
        kels_raw.append(csv_files)
        kels.append(file_names)

    # kels: [[L2Y1S, ...]]
    return kels_raw, kels

def get_csvs():
    path_dataset = './dataset'
    path_data2013 = os.path.join(path_dataset, '2013data')
    data2013 = sorted(os.listdir(path_data2013))

    kels_raw, kels = get_file_structure()
    csv_files = []

    for kels_idx, folder in enumerate(kels_raw):
        csv_year = []
        
        for idx, survey in enumerate(folder):
            exec("%s = pd.read_csv(os.path.join(path_data2013, data2013[kels_idx], survey))" % (kels[kels_idx][idx]))
            exec("csv_year.append(%s)" % (kels[kels_idx][idx]))
        
        csv_files.append(csv_year)

    return csv_files