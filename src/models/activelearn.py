from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

class Active_Learner:
    def __init__(self, model, start_size,end_size,step_size):
        self.model = model
        self.start_size = start_size
        self.end_size = end_size
        self.step_size = step_size

    def fit(self, X, Y):
        X_train, X_unlabeled, Y_train, Y_unlabeled = train_test_split(X, Y, test_size=len(Y)-self.start_size, random_state=42)
        X_train_temp = np.sort(X_train)
        X_train_temp = np.sort(X_train, axis=0)
        np.savetxt('X_train_pre.txt', X_train_temp)
        Y_train_temp = np.sort(Y_train)
        Y_train_temp = np.sort(Y_train, axis=0)
        np.savetxt('Y_train_pre.txt', Y_train_temp)
        while(len(Y_train) <= self.end_size):
            self.model.fit(X_train, Y_train)
            Y_unlabeled_hat = self.model.predict(X_unlabeled)
            low_conf = np.sort(Y_unlabeled_hat, axis=1)
            low_conf = np.diff(low_conf, axis=1)

            lowest_conf_idx = np.flip(np.argsort(low_conf[:,-1]),axis=0)

            #add points of least confidence to training set
            X_train = np.concatenate((X_train,X_unlabeled[lowest_conf_idx[:self.step_size]]),axis=0)
            Y_train = np.concatenate((Y_train,Y_unlabeled[lowest_conf_idx[:self.step_size]]),axis=0)

            #remove these points from "unlabeled" set
            mask = np.ones(len(Y_unlabeled), dtype=bool)
            mask[lowest_conf_idx[0:self.step_size]] = False
            Y_unlabeled = Y_unlabeled[mask]
            X_unlabeled = X_unlabeled[mask]

        X_train_temp = np.sort(X_train)
        X_train_temp = np.sort(X_train, axis=0)
        np.savetxt('X_train.txt', X_train_temp)
        Y_train_temp = np.sort(Y_train)
        Y_train_temp = np.sort(Y_train, axis=0)
        np.savetxt('Y_train.txt', Y_train_temp)
        return self.model

    def predict(self,X):
        return self.model.predict(X)
