import DataSet;
import numpy as np;
from sklearn.datasets import load_breast_cancer;
from sklearn.ensemble import GradientBoostingClassifier;
import matplotlib.pyplot as plt;
import Boundary;

def plot_multi_class_logistic_regression(X,y,dict_names=None, colors= None,  title =None):
    '''
    Draw the multi class samples of 2 features
    :param X: X 2 ndarray (m,2),
    :param y: vector (m,)
    :param dict_names: dict of values of y and names
    :return: None
    '''
    if not colors:
        colors_for_points = ['green','grey', 'orange', 'brown']
    else:
         colors_for_points = colors

    y_unique = list(set(y))

    for i in range (len(y_unique)):
        ind = y == y_unique[i] # vector

        if dict_names:
            plt.scatter(X[ind,0], X[ind,1], c=colors_for_points[i], s=40, label=dict_names[y_unique[i]],edgecolor='black', alpha=.7)
        else:
            plt.scatter(X[ind, 0], X[ind, 1], s=40, c=colors_for_points [i], edgecolor = 'black', alpha = 0.7)
    if title:
        plt.title(title)

    if dict_names:
        plt.legend(frameon=True)


def Model(X_train, X_test, y_train, y_test):
    clf = GradientBoostingClassifier(learning_rate=0.01, max_depth=3).fit(X_train, y_train);
    print("train accuracy= {:.3%}".format(clf.score (X_train, y_train)))
    print("test accuracy= {:.3%}".format(clf.score (X_test, y_test)))
    return clf;

def boundary():
    X, y, labels, features = DataSet.get_dataFor_Boundary();
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    clf = Model(X_train, X_test, y_train, y_test);
    Boundary.plot_labeled_decision_regions(X, y, clf);
    return 0;

def Start():
    X, y, labels, features = DataSet.get_data()
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    bs_dict = dict(zip(np.unique(y), labels));
    plt.figure()
    plot_multi_class_logistic_regression(X, y, dict_names=bs_dict);
    plt.show();
    clf = Model(X_train, X_test, y_train, y_test);
    boundary();

Start();

