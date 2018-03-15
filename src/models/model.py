from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

class Model(object):
    def __init__(self, type, num_neighbors=None, sample='Random', PCA=False):
        if(type == 'KNN'):
            assert (num_neighbors), 'Specify a num_neighbors'
            self.classifier = KNN(num_neighbors) #change probability, right now one probability
        elif(type == 'RF'):
            self.classifier = RandomForestClassifier()
        elif(type == 'LR'):
            self.classifier = LogisticRegression()
        else:
            self.classifier = SVC(decision_function_shape='ovr', probability=True, kernel='linear')
        self.type = type
        self.trained = False
        self.trainedSize = 0
        self.sample = sample
        self.PCA = PCA

    def fit(self, X, Y):
        self.is_fit = True
        self.trainedSize = len(Y)
        self.classifier.fit(X,Y)

    def predict(self, X, proba=True):
        assert (self.is_fit), 'You have not fit the model'
        if(proba):
            return self.classifier.predict_proba(X)
        else:
            return self.classifier.predict(X)

    def test(self, X, Y, fname=None):
        assert (self.is_fit), 'You have not fit the model'
        report = str(self.sample) + " " + str(self.type) + " trained on " + str(self.trainedSize) + " datapoints"
        if(self.PCA):
            report += " (PCA):\n"
        else:
            report += ":\n"
        report += str(classification_report(Y,self.predict(X, proba=False))) + "\n"
        if(fname):
            with open(fname, "a") as myfile:
                myfile.write(report)
        else:
            print(report)

    def activeLearn(self, X, Y, start_size, end_size, step_size):
        X_train, X_unlabeled, Y_train, Y_unlabeled = train_test_split(X, Y, test_size=len(Y)-start_size)

        self.fit(X_train, Y_train)

        while(len(Y_train) < end_size):
            Y_unlabeled_hat = self.predict(X_unlabeled)

            # sort by highest probabilities, and then take difference to find pts
            # model feels strongly are two different classes
            low_conf = np.sort(Y_unlabeled_hat, axis=1)
            low_conf = np.diff(low_conf, axis=1)
            lowest_conf_idx = np.argsort(low_conf[:,-1])

            #add points of least confidence to training set
            X_train = np.concatenate((X_train,X_unlabeled[lowest_conf_idx[:step_size]]),axis=0)
            Y_train = np.concatenate((Y_train,Y_unlabeled[lowest_conf_idx[:step_size]]),axis=0)

            # fit model with new points
            self.fit(X_train, Y_train)

            #remove these points from "unlabeled" set
            mask = np.ones(len(Y_unlabeled), dtype=bool)
            mask[lowest_conf_idx[0:step_size]] = False
            Y_unlabeled = Y_unlabeled[mask]
            X_unlabeled = X_unlabeled[mask]

        return

    def test_metric(self, X_test, Y_test, f1=True, avg='weighted'):
        if(f1):
            Y_hat = self.predict(X_test, proba=False)
            return(f1_score(Y_test, Y_hat, average=avg))


    def save(self, filename):
        with open(filename, 'wb') as ofile:
            pickle.dump(self.clf, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as ifile:
            self.clf = pickle.load(ifile)
