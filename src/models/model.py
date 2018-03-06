from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import pickle

class Model(object):
    def __init__(self, type, num_neighbors=0):
        if(type == 'KNN'):
            self.classifier = KNN(num_neighbors)
        else:
            self.classifier = SVC(probability=True)
        self.type = type

    def fit(self, X, Y):
        self.classifier.fit(X,Y)

    def predict(self, X, proba=True):
        if(proba):
            return self.classifier.predict_proba(X)
        else:
            return self.classifier.predict(X)

    def save(self, filename):
        with open(filename, 'wb') as ofile:
            pickle.dump(self.clf, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as ifile:
            self.clf = pickle.load(ifile)
