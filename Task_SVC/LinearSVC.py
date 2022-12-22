from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt;
import pandas as pd;
import visual;

def GetData():
    iris = load_iris();
    X, y, labels, features = iris.data, iris.target, iris.target_names, iris.feature_names;
    return X, y, labels, features;

def Model(X, y):
    clf = LinearSVC(C=1).fit(X, y);
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
    print("train accuracy= {:.3%}".format(clf.score(X, y)))
    print('b = {}\nw = {}'.format(clf.intercept_, clf.coef_));
    features_dict = {k: v for k, v in enumerate(labels)}
    visual.plot_multi_class_logistic_regression(X, y, dict_names=features_dict);
    plt.show()
    Boundary();
    return 0;

Start()
