from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import pandas as pd;

def GetData():
    iris = load_iris();
    X, y, labels, features = iris.data, iris.target, iris.target_names, iris.feature_names;
    return X, y, labels, features;

def Start():
    X, y, labels, features = GetData();
    print(X, y);

Start()