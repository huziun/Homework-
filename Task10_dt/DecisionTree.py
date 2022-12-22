from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import os;
import graphviz;
import Boundary;
import DataSet;

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

def LearnTree(X_train, X_test, y_train, y_test):
    max_depth = 5
    clf = DecisionTreeClassifier(
        criterion='entropy',
        random_state=20,
        max_depth=max_depth,
    ).fit(X_train, y_train)
    print("train accuracy= {:.3%}".format(clf.score(X_train, y_train)))
    print("test accuracy= {:.3%}".format(clf.score(X_test, y_test)))
    return clf;

def Visualization(clf, labels, features):
    graph_viz = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names=labels, filled=True)
    graph = graphviz.Source(graph_viz)
    graph.view(cleanup=True)
    return 0;

def PlotBoundary(clf):
    X, y, labels, features = DataSet.get_dataFor_Boundary();
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    clf = LearnTree(X_train, X_test, y_train, y_test);
    Boundary.plot_labeled_decision_regions(X, y, clf);
    return 0;

def Start():
    X, y, labels, features = DataSet.get_data();
    X_train, X_test, y_train, y_test = DataSet.Split(X, y);
    clf = LearnTree(X_train, X_test, y_train, y_test);
    Visualization(clf, labels, features);
    PlotBoundary(clf);
    return 0;

Start()