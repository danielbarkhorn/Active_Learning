from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle

class Model(object):
    def __init__(self, type, num_neighbors=None, sample='Random'):
        if(type == 'KNN'):
            assert (not num_neighbors), 'Specify a num_neighbors'
            self.classifier = KNN(num_neighbors)
        else:
            self.classifier = SVC(probability=True)
        self.type = type
        self.trained = False
        self.trainedSize = 0
        self.sample = sample

    def fit(self, X, Y):
        self.trainedSize = len(X)
        self.classifier.fit(X,Y)

    def predict(self, X, proba=True):
        assert (not self.fit), 'You have not fit the model'
        if(proba):
            return self.classifier.predict_proba(X)
        else:
            return self.classifier.predict(X)

    def test(self, X, Y, fname=None):
        assert (not self.fit), 'You have not fit the model'
        report = str(self.sample) + " " + str(self.type) + " trained on " + str(self.trainedSize) + " datapoints:\n"
        report += str(classification_report(Y,self.predict(X, proba=False))) + "\n"
        if(fname):
            with open(fname, "a") as myfile:
                myfile.write(report)
        else:
            print(report)

    def save(self, filename):
        with open(filename, 'wb') as ofile:
            pickle.dump(self.clf, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as ifile:
            self.clf = pickle.load(ifile)
