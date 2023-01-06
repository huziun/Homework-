from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt;
import numpy as np;
import seaborn as sns

def GetData():
    data_set = load_breast_cancer()
    df = pd.DataFrame(data_set.data, columns=data_set.feature_names)
    df['target'] = data_set.target
    return df, data_set

def GetX_Y(df, data):
    Y = df['target'];
    X = data.data
    X = X.reshape(-1, 30)
    return X, Y

def Scaler_data(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x

def Model():
    clf = PCA(n_components=2)
    clf = clf.fit_transform(x)
    return clf

def DrawModel(df_breast_cancer):
    sns.set()

    sns.lmplot(
        x='features 1',
        y='features 2',
        data=df_breast_cancer,
        fit_reg=False,
        legend=True
    )

    plt.title('2D PCA Graph Breast cancer')
    plt.show()

df, data_set = GetData();
X, Y = GetX_Y(df, data_set);
x = Scaler_data(X)

clf = Model();

df_breast_cancer = pd.DataFrame(data = clf, columns = ['features 1', 'features 2'])
DrawModel(df_breast_cancer)



