from sklearn.datasets import load_iris;
from sklearn.svm import SVC
import visual;
import pandas as pd;

degree = 3

def GetData():
    iris = load_iris();
    X, y, labels, features = iris.data, iris.target, iris.target_names, iris.feature_names;
    return X, y, labels, features;

def Model(X, y):
    clf = SVC(kernel='poly').fit(X, y)  # default=3)
    print("train accuracy= {:.3%}".format(clf.score(X, y)))
    return clf;

def GetTwoFiatures():
    iris = load_iris();
    df_cancer = pd.DataFrame(iris.data, columns=iris.feature_names)
    y, labels, features = iris.target, iris.target_names, iris.feature_names;
    X = df_cancer[['petal length (cm)', 'petal width (cm)']];
    y = pd.Series(y);
    return X, y, labels, features;

def Boundary():
    X, y, labels, features = GetTwoFiatures();
    clf = Model(X, y);
    visual.plot_labeled_decision_regions(X, y, clf);
    return 0;

def Start():
    X, y, labels, features = GetData();
    clf = Model(X, y);
    Boundary();
    return 0;

Start();