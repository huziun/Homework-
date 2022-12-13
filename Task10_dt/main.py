from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

def get_data():
    cancer = load_breast_cancer();
    X, y, labels, features = cancer.data, cancer.target, cancer.target_names, cancer.feature_names;
    print('X.shape= ', X.shape)
    print('y.shape= ', y.shape)
    print('labels:', labels);
    print('features:', features);
    return X, y, labels, features;

def Split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0);
    print('X_train.shape= ', X_train.shape)
    print('X_test.shape= ', X_test.shape)
    print('y_train.shape= ', y_train.shape)
    print('y_test.shape= ', y_test.shape)
    return X_train, X_test, y_train, y_test;

def LearnTree(X_train, X_test, y_train, y_test):
    max_depth = 5
    clf = DecisionTreeClassifier(
        criterion='entropy',
        random_state=20,
        max_depth=max_depth,
        #     max_leaf_nodes=4,
    ).fit(X_train, y_train)
    print("train accuracy= {:.3%}".format(clf.score(X_train, y_train)))
    print("test accuracy= {:.3%}".format(clf.score(X_test, y_test)))
    return clf;

def Visualization(clf, labels, features):
    graph_viz = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names=labels, filled=True)
    graph = graphviz.Source(graph_viz)
    graph.view(cleanup=True)
    return 0;

def Start():
    X, y, labels, features = get_data();
    X_train, X_test, y_train, y_test = Split(X, y);
    clf = LearnTree(X_train, X_test, y_train, y_test);
    #Visualization(clf, labels, features);
    return 0;

Start()