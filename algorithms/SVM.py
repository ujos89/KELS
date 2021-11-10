import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import get_csvs, preprocessing_stu, count_arr
from sklearn.svm import SVC
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
X, y = df_input, df_label["L2Y6_E_CS"]
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

y_test = y_test.to_numpy()
print("train size:", X_train.shape)
print("test size:", X_test.shape)
print("test distirbution:", count_arr(y_test))
print()

## SVM
# support vector machine wiht poly kernel
SVM_poly = SVC(kernel = 'poly')
SVM_poly.fit(X_train, y_train)
y_pred = SVM_poly.predict(X_test)
# get accuracy_score
print("poly accuracy:", accuracy_score(y_pred, y_test)*100,"%")
print()

# support vector machine wiht rbf kernel
SVM_rbf = SVC(kernel = 'rbf')
SVM_rbf.fit(X_train, y_train)
y_pred = SVM_rbf.predict(X_test)
# get accuracy_score
# print("y_pred distribution", count_arr(y_pred))
# print("rbf accuracy:", accuracy_score(y_pred, y_test)*100,"%")
# print()
# print("confusion matrix")
# print(confusion_matrix(y_test, y_pred))
# print()
# print("classificaion report")
# print(classification_report(y_test, y_pred))
# print()

# support vector machine wiht linear kernel
SVM_linear = SVC(kernel = 'linear')
SVM_linear.fit(X_train, y_train)
y_pred = SVM_linear.predict(X_test)
# get accuracy_score
print("linear accuracy:", accuracy_score(y_pred, y_test)*100,"%")
print("confusion matrix")
print(confusion_matrix(y_test, y_pred))
print()