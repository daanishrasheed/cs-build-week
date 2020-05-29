import numpy as np
import operator
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


def euclidean(x, y):
    return np.sqrt(np.sum((x-y)**2))

class KNN():

    def __init__(self, k = 3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.Y_train = y_train
    
    def predict(self, X_test):
        pred = []
        for i in range(len(X_test)):
            d = np.array([euclidean(X_test[i], x_t) for x_t in self.X_train])
            d_sorted = d.argsort()[:self.k]
            neighbors = {}
            for i in d_sorted:
                if self.Y_train[i] in neighbors:
                    neighbors[self.Y_train[i]] += 1
                else:
                    neighbors[self.Y_train[i]] = 1
            neighbors_sorted = sorted(neighbors.items(), key = operator.itemgetter(1), reverse = True)
            pred.append(neighbors_sorted[0][0])
        return pred

 
iris = load_iris()
x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

clf = KNN(k=5)
clf.fit(X_train, y_train)
 
predictions = clf.predict(X_test)
 
print('Accuracy:', accuracy_score(y_test, predictions))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
h = knn.predict(X_test)
print('Accuracy for sklearn:', accuracy_score(y_test, h))