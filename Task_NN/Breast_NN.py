from sklearn.datasets import load_breast_cancer;
import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn.neural_network import MLPClassifier;
from sklearn.preprocessing import StandardScaler

def GetData():
    cancer = load_breast_cancer();
    X, y, labels, features = cancer.data, cancer.target, cancer.target_names, cancer.feature_names;
    return X, y, labels, features;

def Model(X_train, X_test, y_train, y_test,  hidden_layer_sizes=(100,), alpha = 0.0001):
    # YOUR_CODE.  Preproces data, train classifier and evaluate the perfromance on train and test sets
    # START_CODE
    clf = MLPClassifier(  # default=(100,)
        solver='lbfgs',
        max_iter=1000,  # default=200
        hidden_layer_sizes = hidden_layer_sizes,
        alpha = alpha
    ).fit(X_train, y_train)

    print("train accuracy= {:.3%}".format(clf.score(X_train, y_train)));
    print("test accuracy= {:.3%}".format(clf.score(X_train, y_train)));
    # END_CODE

    return clf;

def plot_data_logistic_regression(X, y, label, legend_loc= None, title= None):
    '''
    :param X: 2 dimensional ndarray
    :param y:  1 dimensional ndarray. Use y.ravel() if necessary
    :return:
    '''

    positive_indices = (y == 1)
    negative_indices = (y == 0)
#     import matplotlib as mpl
    colors_for_points = ['grey', 'orange'] # neg/pos

    plt.scatter(X[negative_indices][:,0], X[negative_indices][:,1], s=40, c=colors_for_points [0], edgecolor = 'black', label=label[0], alpha = 0.7)
    plt.scatter(X[positive_indices][:,0], X[positive_indices][:,1], s=40, c=colors_for_points [1], edgecolor = 'black',label=label[1], alpha = 0.7)
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.show();
    return 0;

def ScaledData(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled;

def Start():
    X, y, labels, features = GetData();
    X_train, X_test, y_train, y_test = train_test_split(X, y);
    clf = Model(X_train, X_test, y_train, y_test);
    print("After change hidden layer sizes and alpha:");
    X_train, X_test = ScaledData(X_train, X_test);
    clf = Model(X_train, X_test, y_train, y_test,  hidden_layer_sizes=(5,100), alpha=0.0003);
    plot_data_logistic_regression(X, y, labels);

Start()


