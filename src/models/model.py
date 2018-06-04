from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pickle
import os

class Model(object):
    def __init__(self, type, num_neighbors=None, sample='Random', PCA=False, name=None):
        if(type == 'KNN'):
            assert (num_neighbors), 'Specify a num_neighbors'
            self.classifier = KNN(num_neighbors) #change probability, right now one probability
        elif(type == 'RF'):
            self.classifier = RandomForestClassifier()
        elif(type == 'LR'):
            self.classifier = LogisticRegression()
        elif(type == 'NN'):
            assert (name), 'Name your net'
            self.name = name
            self.classifier = 'NN'
            self.init_NN()
        else:
            self.classifier = SVC(decision_function_shape='ovr', probability=True, kernel='linear')
            type = 'SVM'
        self.type = type
        self.trained = False
        self.trainedSize = 0
        self.sample = sample
        self.PCA = PCA
        self.name = name

    def fit(self, X, Y):
        self.is_fit = True
        self.trainedSize += len(Y)
        if self.classifier == 'NN':
            self.fit_NN(X, Y)
        else:
            self.classifier.fit(X,Y)

    def fit_NN(self, X, Y):
        self.is_fit = True
        X /= 256
        batchsize = 50

        with tf.Session() as sess:
            self.saver.restore(sess, "NN/"+self.name+".ckpt")
            for i in range(0, len(X), batchsize):
                # self.optimizer.eval(feed_dict={self.x: X[i:i+batchsize], self.y: self.one_hot_encode(Y[i:i+batchsize])})
                # self.cross_entropy.eval(feed_dict={self.x: X[i:i+batchsize], self.y: self.one_hot_encode(Y[i:i+batchsize])})
                _, c = sess.run([self.optimizer, self.cross_entropy], feed_dict={self.x: X[i:i+batchsize], self.y: self.one_hot_encode(Y[i:i+batchsize])})
            self.saver.save(sess, "NN/"+self.name+".ckpt")

    def predict_NN(self, X, proba=True):
        assert (self.is_fit), 'You have not fit the model'
        X /= 256
        with tf.Session() as sess:
            self.saver.restore(sess, "NN/"+self.name+".ckpt")
            yHat = self.y_.eval(feed_dict={self.x: X}) #sess.run(self.y_, feed_dict={self.x: X})
            self.saver.save(sess, "NN/"+self.name+".ckpt")
        if proba:
            return yHat
        else:
            return np.argmax(yHat, axis=1)


    def predict(self, X, proba=True):
        assert (self.is_fit), 'You have not fit the model'
        if self.type == 'NN':
            return self.predict_NN(X, proba=proba)
        else:
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

    #TODO:
    # Implement new active learning method, with 'boosting' liek choices
    # IE include randomly chosen points, add them multiple times if the model
    # was very confident in its classificaiton, but was wrong
    def activeLearn(self, X, Y, start_size, end_size, step_size, SVM_D=False, random_size=0, outlier_size=0):
        X_train, X_unlabeled, Y_train, Y_unlabeled = train_test_split(X, Y, test_size=len(Y)-start_size)
        self.fit(X_train, Y_train)
        while(len(Y_train) < end_size):
            if SVM_D and self.type=='SVM':
                hyperplane_dists = self.classifier.decision_function(X_unlabeled)

                # sort by closeness to decision boundary. some can be negative so absval
                # https://stackoverflow.com/questions/46820154/negative-decision-function-values
                low_conf = np.sort(np.abs(hyperplane_dists), axis=1)
                lowest_conf_idx = np.argsort(low_conf[:,0])

            else:
                Y_unlabeled_hat = self.predict(X_unlabeled)

                # sort by highest probabilities, and then take difference to find pts
                # model feels strongly are two different classes
                low_conf = np.sort(Y_unlabeled_hat, axis=1)
                lowest_outliers = np.argsort(low_conf[:,-1])
                low_conf = np.diff(low_conf, axis=1)
                lowest_conf_idx = np.argsort(low_conf[:,-1])

            random_points_idx = []
            if random_size:
                random_points_idx = np.random.choice(np.arange(step_size, len(lowest_conf_idx)), random_size, replace=False)

                random_points_x = [X_unlabeled[i] for i in random_points_idx]
                random_points_y = [Y_unlabeled[i] for i in random_points_idx]


            boosted_points = []
            for ind in random_points_idx:
                Y_hat = np.argmax(Y_unlabeled_hat[ind])
                if Y_unlabeled[ind] != Y_hat:
                    # we incorrectly predicted the random point
                    boosted_points += [ind]*int(1 + Y_unlabeled_hat[ind][Y_hat]//.25)
            if boosted_points:
                random_points_idx = np.concatenate((random_points_idx, boosted_points))

            outliers = []
            if outlier_size:
                outliers = lowest_outliers[-outlier_size:]

            chosen_idx = np.concatenate((lowest_conf_idx[:(step_size - random_size)], random_points_idx, outliers)).astype(int)

            #add chosen points to training set
            X_train = np.concatenate((X_train,X_unlabeled[chosen_idx]),axis=0)
            Y_train = np.concatenate((Y_train,Y_unlabeled[chosen_idx]),axis=0)

            # fit model with new points
            self.fit(X_train, Y_train)

            #remove these points from "unlabeled" set
            mask = np.ones(len(Y_unlabeled), dtype=bool)
            mask[chosen_idx] = False
            Y_unlabeled = Y_unlabeled[mask]
            X_unlabeled = X_unlabeled[mask]

        return

    def test_metric(self, X_test, Y_test, f1=True, avg='weighted'):
        if(f1):
            Y_hat = self.predict(X_test, proba=False)
            return(f1_score(Y_test, Y_hat, average=avg))

    def init_NN(self):
        tf.reset_default_graph()

        self.learning_rate = 0.5

        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        # first layer weights and bias
        self.W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
        self.b1 = tf.Variable(tf.random_normal([300]), name='b1')
        # second layer weight and bias
        self.W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
        self.b2 = tf.Variable(tf.random_normal([10]), name='b2')

        self.hidden_out = tf.add(tf.matmul(self.x, self.W1), self.b1)
        self.hidden_out = tf.nn.relu(self.hidden_out)

        self.y_ = tf.nn.softmax(tf.add(tf.matmul(self.hidden_out, self.W2), self.b2))

        self.y_clipped = tf.clip_by_value(self.y_, 1e-10, 0.9999999)
        self.cross_entropy = -tf.reduce_mean(tf.reduce_sum(self.y * tf.log(self.y_clipped)
                                                      + (1 - self.y) * tf.log(1 - self.y_clipped), axis=1))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

        self.init_op = tf.global_variables_initializer()

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(self.init_op)
            self.saver.save(sess, "NN/"+self.name+".ckpt")

    def save(self, filename):
        with open(filename, 'wb') as ofile:
            pickle.dump(self.clf, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as ifile:
            self.clf = pickle.load(ifile)

    def one_hot_encode(self, y_original):
        y_encoded = np.array(np.zeros((y_original.shape[0], 10)))

        i = 0
        for num in y_original:
            y_encoded[i][int(num)] = 1
            i = i + 1

        return y_encoded
