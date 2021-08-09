import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from utils import get_csvs, preprocessing_stu

path = './dataset'
path_data2013 = os.path.join(path, '2013data')
raw_csvs = get_csvs()

L2Y1S = raw_csvs[0][0]
# L2Y1P = raw_csvs[0][1]

df_input, df_label = preprocessing_stu(L2Y1S)
print("input shape:", df_input.shape)
# print(df_input.head())
print()

def pca_visualization(df_input, df_label, label="L2Y1_E_CS"):
    df = pd.merge(df_input, df_label[label], left_index=True, right_index=True, how='left')
    df = df.rename(columns={0:'pc1', 1:'pc2'})
    grouped = df.groupby(label)

    plt.rcParams['figure.figsize'] = [10, 8]
    fig, ax = plt.subplots()

    for name, group in grouped:
        ax.plot(group.pc1, group.pc2, marker='o', linestyle='', label=name)

    ax.legend(fontsize=12, loc='upper left') # legend position

    plt.title('Scatter Plot of PCA with 2 components', fontsize=20)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.show()



# pca with 2 components (for visualization)
pca = PCA(n_components = 2)
df_pca = pd.DataFrame(pca.fit_transform(df_input), index = df_input.index)
print("PCA with 2 components")
# print(df_pca.head())
print(" explained variance ratio :", pca.explained_variance_ratio_)
print(" sum of explained variance ratio :", sum(pca.explained_variance_ratio_)*100, "%")
print()

# find valid number of components
pca = PCA(n_components=.95)
pca_95 = pd.DataFrame(pca.fit_transform(df_input), index = df_input.index)
# print(pca_95.head())
print("valid number of componets: ",pca_95.shape[1])
print()

pca_visualization(df_pca, df_label)