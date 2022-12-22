from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split;
#import load_breast_cancer and get the X_cancer, y_cancer
from sklearn.datasets import load_breast_cancer;
import matplotlib.pyplot as plt;
import pandas as pd;
import visual;

def GetData():
     data = load_breast_cancer();
     df = pd.DataFrame(data.data, columns=data.feature_names);
     df['target'] = data.target;
     return df, data;

def GetX_Y(df, data):
     Y = df['target'];
     X = data.data;
     X = X.reshape(-1,30);
     return X, Y;

def Split(X, y):
     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0);
     return X_train,  X_test, y_train, y_test;

df, data = GetData();
X_cancer, y_cancer = GetX_Y(df, data);
X_train, X_test, y_train, y_test = Split(X_cancer, y_cancer);
clf = LogisticRegression(max_iter=1000, C=1).fit(X_train, y_train);

def Visual_Boundary():
     visual.plot_data_logistic_regression(X_cancer, y_cancer);
     plt.show();
     return 0;

print('X_cancer.shape= {}'.format(X_cancer.shape));
print('\nBreast cancer dataset');
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)));
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)));
Visual_Boundary();
