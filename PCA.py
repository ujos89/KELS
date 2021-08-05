import numpy as np
import pandas as pd
import os

from utils import get_csvs, preprocessing_stu

path = './dataset'
path_data2013 = os.path.join(path, '2013data')

raw_csvs = get_csvs()

L2Y1S = raw_csvs[0][0]
# L2Y1P = raw_csvs[0][1]

df_input, df_label = preprocessing_stu(L2Y1S)