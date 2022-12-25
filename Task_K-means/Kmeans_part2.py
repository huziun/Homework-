from sklearn.datasets import make_blobs
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.cluster import AgglomerativeClustering

colors = np.array(['green', 'red', 'yellow', 'brown', 'black', 'purple', 'blue', 'pink', 'orange', 'grey']);
n_cluster = 4
X, y = make_blobs(n_samples = 500,
                      n_features = 2,
                      centers = 5,
                      cluster_std = 0.6,
                      random_state = 0);

x = X[:, 0]
y_train = X[:, 1]

plt.figure()
plt.scatter(x, y_train);

def Aglom_singel():

    clf = AgglomerativeClustering(n_clusters=n_cluster, linkage= 'single')
    predicted = clf.fit_predict(X)
    plt.figure(figsize=(6,6))
    plt.scatter(x, y_train, c= colors[predicted], marker= '.')

    plt.title('sklearn kmeans single')
    plt.show();

def Aglomer_complete():

    clf = AgglomerativeClustering(n_clusters=n_cluster, linkage='complete')
    predicted = clf.fit_predict(X)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y_train, c=colors[predicted], marker='.')

    plt.title('sklearn kmeans complete')
    plt.show();

Aglom_singel();
Aglomer_complete();