import os
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from algorithms.DT import Decision_Tree #Decision tree
from algorithms.SVM import SVM #Support Vector Machine
from algorithms.ExtraTrees import Extra_Trees #Extra trees classifier
from algorithms.GradientBoosting import Grad_Boost
from algorithms.KNN import KNN #k-nearest neighbors
from algorithms.RandomForest import RandomForest

path_data ='./preprocessed/prepared/drop/fill/'

file_names = os.listdir(path_data)
flag = False

for file_name in file_names:
    df_ = pd.read_pickle(os.path.join(path_data, file_name))
    
    if not file_name.startswith('label'):    
        if not flag:
            flag = True
            input = df_
        elif flag:
            input = pd.concat([input, df_], axis=1)

    else:
        label = df_
        
sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
X, y = input, label["L2Y6_K_CS"]

for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

y_test = y_test.to_numpy()
print("train size:", X_train.shape)
print("test size:", X_test.shape)

# Decision_Tree(X_train, X_test, y_train, y_test)
# SVM(X_train, X_test, y_train, y_test, kernel='rbf')
# Extra_Trees(X_train, X_test, y_train, y_test)
# Grad_Boost(X_train, X_test, y_train, y_test)
# KNN(X_train, X_test, y_train, y_test)
RandomForest(X_train, X_test, y_train, y_test)