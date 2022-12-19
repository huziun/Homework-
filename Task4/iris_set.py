import pandas as pd;
from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.metrics import confusion_matrix

def GetData():
    iris = load_iris();
    return iris;

def GetX_Y(iris):
    X, y, labels, feature_names = iris.data, iris.target, iris.target_names, iris['feature_names'];
    return X, y, labels, feature_names;

def DF_Iris(X, y, labels, feature_names):
    df_iris = pd.DataFrame(X, columns=feature_names);
    df_iris['label'] = y;
    features_dict = {k:v for k,v in enumerate(labels)}
    df_iris['label_names'] = df_iris.label.apply(lambda x: features_dict[x]);
    return df_iris;

def Split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25);
    return X_train, X_test, y_train, y_test;

def Normalization(X_train, X_test):
    scaler = StandardScaler();
    X_train_scaled = scaler.fit_transform(X_train);
    X_test_scaled = scaler.transform(X_test);

    return X_train_scaled, X_test_scaled;

def Classification(X_train_scaled, X_test_scaled, y_train, y_test):
    max_score_test = 0;
    k_best = 1;

    for item in range(1, 40, 1):
        knn_Classifier = KNeighborsClassifier(n_neighbors=item);
        knn_Classifier = knn_Classifier.fit(X_train_scaled, y_train);
        score = knn_Classifier.score(X_test_scaled, y_test);

        if score > max_score_test:
            max_score_test = score;
            k_best = item;

    predict = knn_Classifier.predict(X_test_scaled);
    print("confusion_matrix:\n", confusion_matrix(y_test, predict));
    print("K best = ", k_best);
    print("Score Test: ", max_score_test);
    return knn_Classifier;

def Start():
    iris = GetData();
    X, y, labels, feature_names = GetX_Y(iris);
    df_iris = DF_Iris(X, y, labels, feature_names);
    X_train, X_test, y_train, y_test = Split(X, y);
    X_train_scaled, X_test_scaled = Normalization(X_train, X_test);
    Classification(X_train_scaled, X_test_scaled, y_train, y_test);
    return 0;
