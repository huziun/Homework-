from sklearn.datasets import load_breast_cancer;
import pandas as pd;
from sklearn.preprocessing import StandardScaler;
import matplotlib.pyplot as plt;
import numpy as np;

def GetData():
    data_set = load_breast_cancer();
    df = pd.DataFrame(data_set.data, columns=data_set.feature_names);
    df['target'] = data_set.target;
    return df, data_set;

def GetX_Y(df, data):
    Y = df['target'];
    X = data.data;
    X = X.reshape(-1, 30);
    return X, Y;

def Scaled(X):
    scaler = StandardScaler();
    X_scaled= scaler.fit_transform(X);
    return X_scaled;

def Plot(X):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=40, edgecolor='black', label='negative', alpha=0.7);
    plt.show();
    return 0;

def Cov_Matrix():
    cov_matrix = np.cov(X.T, ddof=0) # ddof=0 will return the simple average
    print('numpy cov_matrix:\n', cov_matrix);
    return cov_matrix;

def Plot_vectors(X, v):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=40, edgecolor='black', label='negative', alpha=0.7)

    v0 = v[:, 0]  # Note: every columns is vector
    v1 = v[:, 1]

    plt.plot([0, v0[0] * 3], [0, v0[1] * 3], '-', c='r')
    plt.plot([0, v1[0] * 3], [0, v1[1] * 3], '-', c='g');
    plt.show();
    return 0;

def TransformData(X, v):
    k = 1
    U_reduce = v[:, :k]
    new_data = X @ U_reduce
    print(new_data.shape);
    return new_data, U_reduce;

def Decompres_data(U_reduce, new_data):
    X_approximate = new_data @ U_reduce.T
    return X_approximate;

def comress_score(X, X_approximate):
    return np.sum(np.apply_along_axis (np.linalg.norm,1, (X-X_approximate))**2)/\
        np.sum(np.apply_along_axis (np.linalg.norm,1, X)**2)

df, data_set = GetData();
X, y = GetX_Y(df, data_set);
X = Scaled(X);
Plot(X);
cov_matrix = Cov_Matrix();
w, v = np.linalg.eig(cov_matrix)
Plot_vectors(X, v);
new_data, U_reduce = TransformData(X, v)
X_approximate = Decompres_data(U_reduce, new_data);
print('Compress score= {0:.3f}'.format(comress_score(X, X_approximate)))

plt.figure()
plt.scatter(X_approximate[:,0], X_approximate[:,1], s= 40, edgecolor = 'black', label='negative', alpha = 0.7)
plt.show();