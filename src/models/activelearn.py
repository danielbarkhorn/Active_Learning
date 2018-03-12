from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

class Active_Learner:
    def __init__(self, model, start_size=0.1,end_size=0.25,step_size=0.01):
        self.model = model
        self.start_size = start_size
        self.end_size = end_size
        self.step_size = step_size

    def fit(self, X, Y):
        X_train, X_unlabeled, Y_train, Y_unlabeled = train_test_split(X, Y, test_size=1-self.start_size, random_state=42)

        while(len(Y_train) <= len(Y)*self.end_size):
            self.model.fit(X_train,Y_train)
            Y_unlabeled_hat = self.model.predict(X_unlabeled)

            #select points with low confidence (class probabilty close to 0.5)
            lowest_conf_idx = np.flip(np.argsort(abs(0.5-Y_unlabeled_hat[:,0])),axis=0)

            #add points of least confidence to training set
            X_train = np.concatenate((X_train,X_unlabeled[:int(len(Y)*self.step_size)]),axis=0)
            Y_train = np.concatenate((Y_train,Y_unlabeled[:int(len(Y)*self.step_size)]),axis=0)

            #remove these points from "unlabeled" set
            mask = np.ones(len(Y_unlabeled), dtype=bool)
            mask[lowest_conf_idx[0:int(len(Y)*self.step_size)]] = False
            Y_unlabeled = Y_unlabeled[mask]
            X_unlabeled = X_unlabeled[mask]

        return self.model

    def predict(self,X):
        return self.model.predict(X)
