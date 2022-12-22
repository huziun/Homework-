from sklearn.datasets import load_breast_cancer;
from sklearn.model_selection import train_test_split;
import pandas as pd;

def get_data():
    cancer = load_breast_cancer();
    X, y, labels, features = cancer.data, cancer.target, cancer.target_names, cancer.feature_names;
    print('labels:', labels);
    print('features:', features);
    return X, y, labels, features;

def get_dataFor_Boundary():
    cancer = load_breast_cancer();
    df_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    y, labels, features = cancer.target, cancer.target_names, cancer.feature_names;
    X = df_cancer[['mean radius', 'mean concave points']];
    y = pd.Series(y);
    return X, y, labels, features;

def Split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0);
    print('X.shape= ', X.shape)
    print('y.shape= ', y.shape)
    return X_train, X_test, y_train, y_test;