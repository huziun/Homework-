from sklearn.ensemble import RandomForestClassifier;
import Boundary;
import DataSet;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.tree import plot_tree

def Model(X_train, X_test, y_train, y_test):
    max_features_list = [20, 10, 8, 4, 2]
    for i, max_features in enumerate(max_features_list):
        clf = RandomForestClassifier(
            random_state=0,
            max_features=max_features,
        ).fit(X_train, y_train)
        accuracy_train = clf.score(X_train, y_train)
        accuracy_test = clf.score(X_test, y_test)
        print('max_features = {}:\n\t accuracy_train = {:.3%}\n\t accuracy_test = {:.3%}'.format(
            max_features_list[i], accuracy_train, accuracy_test))

    return clf;

def Visualization(clf, labels, features):
    fig = plt.figure(figsize=(15, 10))
    plot_tree(clf.estimators_[0],
              feature_names=features,
              class_names=labels,
              filled=True, impurity=True,
              rounded=True)
    plt.show()
    return 0;

def VisualBoundary():
    X, y, labels, features = DataSet.get_dataFor_Boundary();
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    clf = Model(X_train, X_test, y_train, y_test);
    Boundary.plot_labeled_decision_regions(X, y, clf);
    return 0;

def Start():
    X, y, labels, features = DataSet.get_data();
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    clf = Model(X_train, X_test, y_train, y_test);
    Visualization(clf, labels, features);
    VisualBoundary();
    return 0;

Start()