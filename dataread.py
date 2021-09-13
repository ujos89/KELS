import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import *

# extract data
path = './dataset'
path_data2013 = os.path.join(path, '2013data')
raw_csvs = get_csvs()
L2Y1S = raw_csvs[0][0]
# L2Y1P = raw_csvs[0][1]
L2Y2S = raw_csvs[1][0]
L2Y3S = raw_csvs[2][0]
L2Y4S = raw_csvs[3][0]
L2Y5S = raw_csvs[4][0]

# df_input, df_label = preprocessing_stu(L2Y1S)
df_input, df_label = preprocessing_stu2(L2Y2S)

