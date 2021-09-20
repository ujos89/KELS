import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import *

# extract data
path = './dataset'
path_data2013 = os.path.join(path, '2013data')
raw_csvs = get_csvs()
# L2Y1S = raw_csvs[0][0]
# # L2Y1P = raw_csvs[0][1]
# L2Y2S = raw_csvs[1][0]
# L2Y3S = raw_csvs[2][0]
# L2Y4S = raw_csvs[3][0]
# L2Y5S = raw_csvs[4][0]



for year in range(1,7):
    df_input, df_label = preprocessing_stu(raw_csvs[year-1][0], year)
    print("year: ", year)
    print(df_input.shape)
    print(df_label.shape)
    # print(df_input.head(5))
    print()