import pandas as pd;
from sklearn.svm import SVC;
from sklearn.linear_model import LinearRegression;
from sklearn.linear_model import LogisticRegression;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.model_selection import train_test_split;
from sklearn.model_selection import cross_validate;
import matplotlib.pyplot as plt;
import numpy as np;
import seaborn as sns

#~~~Train.csv~~~

def ReadFile():
    File_name = 'train.csv';
    data_set = pd.read_csv(File_name).set_index('PassengerId');
    data_set.head();
    return data_set;

def DF_set(data_set):
    df = pd.DataFrame(data_set);
    df['Survived'];
    return df.head(30);

def Get_X(data_set):
    X = data_set[['Sex', 'Age', 'Pclass', 'Fare']];
    return X;

def Get_Y(data_set):
    y = data_set[['Survived']];
    return y;

#~~~check data~~~

def NoneData(data_set, column):
    count = 0;
    check_columns = data_set[column];
    count_nan = check_columns.isnull().sum();
    for item in data_set[column]:
        print(item);
        count += 1;
        if item is np.nan:

            print(count);
    return count_nan;

#~~~Convert data~~~

def Convert_Name(data_set):
    data_set.Name = data_set.Name.str.extract('\, ([A-Z][^ ]*\.)', expand=False);
    data_set.Name = lambda x: x if x is not None else 'Mr.';
    return data_set;

def Convert_Sex(data_set):
    data_set['Sex'] = data_set['Sex'].apply(lambda x: 1 if x == 'male' else 0);
    return data_set;

def Convert_Drob(data_set):
    data_set['Age'] = data_set['Age'].apply(lambda x: x * 100 if x < 1 else x);
    return data_set;

#~~~Fill space~~~

def Fill_NA_Age(data_set):
    data_set['Age'].fillna(data_set['Age'].mean(), inplace=True);
    return data_set;

def Fill_NA_Fare(data_set):
    data_set['Fare'].fillna(data_set['Age'].mean(

    ), inplace=True);
    return data_set

def PrepareDATASET_Train():
    data_set = ReadFile();
    data_set = Convert_Drob(data_set);
    data_set = Convert_Name(data_set);
    data_set = Convert_Sex(data_set);
    data_set = Fill_NA_Age(data_set);
    data_set = Fill_NA_Fare(data_set);
    return data_set;

data_set = PrepareDATASET_Train();

def PrepareX_Y(data_set):
    X = Get_X(data_set);
    y = Get_Y(data_set);
    return X, y;

X, y = PrepareX_Y(data_set);

def VisualizeCorrMatrix(df):
    plt.figure()
    correlation = df.corr();
    heatmap = sns.heatmap(correlation, annot= True)
    heatmap.set(xlabel = "X values", ylabel="Survive",title= "CorrelationMatrix")
    plt.show();

#VisualizeCorrMatrix(data_set);

def Split():
    X_train, X_test, y_train, y_test = train_test_split(X, y);
    return X_train, X_test, y_train, y_test;

X_train, X_test, y_train, y_test = Split();

#~~~Train Models~~~

def Model_SVC(X, y):
    clf = SVC(C=2, gamma=3).fit(X, y) # rbf is default
    print("SVM train accuracy= {:.3%}".format(clf.score(X, y)))
    return clf;

def Model_LinerRegresion(X, y):
    clf = LinearRegression().fit(X, y)
    print("LinearRegression train accuracy= {:.3%}".format(clf.score(X, y)));
    return clf;

def Model_LogisticRegression(X, y):
    clf = LogisticRegression(random_state=0, C=2).fit(X, y);
    print("LogisticRegression train accuracy= {:.3%}".format(clf.score(X, y)));
    return clf;

def Model_RandomForest(X, y):
    clf = RandomForestClassifier(max_depth=12, random_state=0).fit(X, y);
    accuracy_train = clf.score(X, y)
    print("RandomForestClassifier train accuracy= {:.3%}".format(clf.score(X, y)));
    return clf;

#~~~Test.csv~~~

def Read_Test():
    File_name = 'test.csv';
    data_set = pd.read_csv(File_name).set_index('PassengerId');
    print(data_set.columns)
    return data_set;

def Train_models(X_train, X_test, y_train, y_test):
    Model_SVC(X_train, y_train);
    Model_LinerRegresion(X_train, y_train);
    Model_LogisticRegression(X_train, y_train)
    Model_RandomForest(X_train, y_train);
    print("~~~Test~~~")
    Model_SVC(X_test, y_test);
    Model_LinerRegresion(X_test, y_test);
    Model_LogisticRegression(X_test, y_test)
    Model_RandomForest(X_test, y_test);
    return 0;

Train_models(X_train, X_test, y_train, y_test);

def PrepareDARASET_Test():
    data_set21 = Read_Test();
    data_set21 = Convert_Name(data_set21);
    data_set21 = Convert_Drob(data_set21);
    data_set21 = Convert_Sex(data_set21);
    data_set21 = Fill_NA_Age(data_set21);
    print()
    print(NoneData(data_set21, 'Fare'))
    data_set21 = Fill_NA_Fare(data_set21);
    return data_set21;

def Get_X_test(data_set21):
    X_test_val = Get_X(data_set21);
    return X_test_val;

def GetPredict(X_test_val):
    clf = Model_RandomForest(X, y);
    y_val = clf.predict(X_test_val);
    X_test_val['Survived'] = y_val;
    return X_test_val;

def Del_Columns(X_test_val):
    del X_test_val['Sex'];
    del X_test_val['Age'];
    del X_test_val['Pclass'];
    del X_test_val['Fare'];
    return X_test_val;

data_set21 = PrepareDARASET_Test();
X_test_val = Get_X_test(data_set21);
X_test_val = GetPredict(X_test_val);
Del_Columns(X_test_val);

X_test_val.to_csv('sudmit.csv');




