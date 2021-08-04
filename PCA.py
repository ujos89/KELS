import numpy as np
import pandas as pd
import os

from utils import get_csvs

path = './dataset'
path_data2013 = os.path.join(path, '2013data')

raw_csvs = get_csvs()
print(len(raw_csvs))