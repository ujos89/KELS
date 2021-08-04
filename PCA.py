import numpy as np
import pandas as pd
import os

from utils import get_file_structure

path = './dataset'
path_data2013 = os.path.join(path, '2013data')

file_structure, kels = get_file_structure()

x='buffalo'    
exec("%s = %d" % (x,2))

