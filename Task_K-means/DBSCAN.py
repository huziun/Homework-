from sklearn.datasets import make_blobs;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn.cluster import DBSCAN;

clusters = 5

X, y = make_blobs(n_samples = 500,
                  n_features = 2,
                  centers = 5,
                  cluster_std = 0.6,
                  random_state = 0)
colors = np.array([plt.cm.Spectral(val)
          for val in np.linspace(0, 1, len(set(y)))])
plt.figure(figsize=(8,6))

plt.scatter(X[:,0], X[:,1], c= colors[y], s= 20)
plt.show()

def apply_db_scan(points, eps):
    dbscan = DBSCAN(eps=eps, min_samples=clusters, ).fit(points)
    predicted = dbscan.labels_
    n_clusters_and_noice = len(np.unique(predicted))
    print ('n clusters and noice ={}\n'.format(n_clusters_and_noice))

    predicted[np.where(predicted == -1)] = n_clusters_and_noice
    return predicted

predicted  =  apply_db_scan(X, eps=1.6) #  1.6 (5) # 1.7(3) # 1.65 (4but...) # 1.5 (8 looks good)
print(predicted)

plt.figure(figsize=(6,6))
colors = np.array(['green','grey', 'orange', 'brown', 'blue', 'yellow'])
colors = np.r_[colors, np.array(['black']*100)]

plt.scatter(X[:,0], X[:,1], c= colors[predicted], s= 40, edgecolor = 'black', label='negative', alpha = 0.7)
plt.show();