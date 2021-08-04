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

