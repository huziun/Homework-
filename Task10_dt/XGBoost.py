from xgboost import XGBClassifier
import matplotlib.pyplot as plt;
import DataSet;
import Boundary;

def plot_data_logistic_regression(X,y,legend_loc= None, title= None):
    '''
    :param X: 2 dimensional ndarray
    :param y:  1 dimensional ndarray. Use y.ravel() if necessary
    :return:
    '''

    positive_indices = (y == 1)
    negative_indices = (y == 0)
#     import matplotlib as mpl
    colors_for_points = ['grey', 'orange'] # neg/pos

    plt.scatter(X[negative_indices][:,0], X[negative_indices][:,1], s=40, c=colors_for_points [0], edgecolor = 'black', label='negative', alpha = 0.7)
    plt.scatter(X[positive_indices][:,0], X[positive_indices][:,1], s=40, c=colors_for_points [1], edgecolor = 'black',label='positive', alpha = 0.7)
    plt.title(title)
    plt.legend(loc=legend_loc)
    return 0;

def Model(X_train, X_test, y_train, y_test):
    clf_xgboost = XGBClassifier(use_label_encoder=False, eval_metric= 'logloss')
    clf_xgboost.fit(X_train, y_train);
    print("train accuracy= {:.3%}".format(clf_xgboost.score(X_train, y_train)))
    print("test accuracy= {:.3%}".format(clf_xgboost.score(X_test, y_test)))
    return clf_xgboost;

def Visual_Boundary():
    X, y, labels, features = DataSet.get_dataFor_Boundary();
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    clf = Model(X_train, X_test, y_train, y_test);
    Boundary.plot_labeled_decision_regions(X, y, clf);
    return 0;

def Start():
    X, y, features, labels = DataSet.get_data();
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    plt.figure()
    plot_data_logistic_regression(X_train, y_train)
    plt.show();
    clf = Model( X_train, X_test, y_train, y_test);
    Visual_Boundary();
    return 0;

Start();
