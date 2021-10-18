import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import get_csvs, preprocessing_stu, plot_2d
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # extract data
# path = './dataset'
# path_data2013 = os.path.join(path, '2013data')
# raw_csvs = get_csvs()
# L2Y1S = raw_csvs[0][0]
# # L2Y1P = raw_csvs[0][1]
# df_input, df_label = preprocessing_stu(L2Y1S)

path_merge = './preprocessed/merge/'
df_input = pd.read_csv(os.path.join(path_merge, 'input_merge.csv'))
df_input = df_input.set_index('L2SID', drop=True)
df_label = pd.read_csv(os.path.join(path_merge, 'label_merge.csv'))
df_label = df_label.set_index('L2SID', drop=True)
df_label = df_label.astype(int)

# train test set split (stratified)
sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
X, y = df_input, df_label["L2Y6_K_CS"]
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

y_test = y_test.to_numpy()
print("train size:", X_train.shape)
print("test size:", X_test.shape)
print()

# Decision tree
dt = DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# accuracy
print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
print()
print("confusion matrix")
print(confusion_matrix(y_test, y_pred))
print()
print("classificaion report")
print(classification_report(y_test, y_pred))