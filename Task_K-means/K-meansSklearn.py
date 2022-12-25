from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt;
import numpy as np;
from sklearn.metrics import accuracy_score;

colors = np.array(['green', 'red', 'yellow', 'brown', 'black', 'purple', 'blue', 'pink', 'orange', 'grey']);

Xc_2,y_= make_classification(n_samples=200,
                                    n_features=2,
                                    n_informative=2,
                                    n_redundant=0,
                                    random_state=0,
                                    n_clusters_per_class=1,
                                    class_sep = 0.8)

x = Xc_2[:, 0]
y = Xc_2[:, 1]

plt.figure()
plt.scatter(Xc_2[:,0], Xc_2[:,1])

def Model(n):
    clf = KMeans(n_clusters=n)
    clf.fit(Xc_2, y_);
    return clf;

def Predict(clf):
    predicted= clf.predict(Xc_2)
    return predicted;

def Draw(predicted, clf):
    plt.scatter(x, y, c=colors[predicted])

    for i, c in enumerate(clf.cluster_centers_):
        plt.plot(c[0], c[1], marker='x', color=colors[i], markersize=14)

    plt.title('sklearn kmeans')
    plt.show();

def Score(predicted):
    score = accuracy_score(y_, predicted);
    print("Score: ", score);
    return score;

def Start():
    array_score = []
    array_nclaster = []
    for number in range(2, 10):
        print("N Clusters: ", number);
        clf = Model(number);
        predict = Predict(clf);
        Draw(predict, clf);
        score = Score(predict);
        array_score.append(score);
        array_nclaster.append(number);


    plt.figure()
    plt.plot(array_nclaster, array_score)
    plt.xlabel('N clusters')
    plt.ylabel('Accuracy score')
    plt.show()



Start();

