import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap
import iris_set;

def Vizual():
    cmap_bold = ListedColormap(['blue','#FFFF00','black','green']);

    np.random.seed = 2021;
    X_D2, y_D2 = make_blobs(n_samples=300, n_features=2, centers=8, cluster_std=1.3, random_state=4);
    y_D2 = y_D2 % 2;
    plt.figure();
    plt.title('Sample binary classification problem with non-linearly separable classes');
    plt.scatter(X_D2[:,0], X_D2[:,1], c=y_D2, marker= 'o', s=30, cmap=cmap_bold);

    plt.show();
    return X_D2, y_D2;

def GetData():
    mb = make_blobs();
    return mb;

def Start():
    X, y = Vizual();
    makeBlobs = GetData();
    print(makeBlobs);
    X_train, X_test, y_train, y_test = iris_set.Split(X, y);
    X_train, X_test = iris_set.Normalization(X_train, X_test);
    iris_set.Classification(X_train, X_test, y_train, y_test);
