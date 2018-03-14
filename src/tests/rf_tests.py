import datetime
import os
import sys
import pickle
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

import numpy as np

from data.dataset import Dataset
from models.model import Model

print(datetime.datetime.now())


mnist_pca = pickle.load(open( "../data/pickled/mnist_data_pca50.p", "rb" ))

mnist_pca_sample = mnist_pca.random_sample(percent=.5) #24 instances

randRFF1s = []
activeRFF1s = []

for _ in range(3):
    #getting test data to use for both models
    (train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.8)

    #make random train data and model
    rand_train_PCA = train_pca.random_sample(size=250)
    rand_RF = Model('RF')
    rand_RF.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
    randRFF1s.append(rand_RF.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True))

    #make active model
    active_RF = Model('RF', sample='Active')
    active_RF.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=150, end_size=250, step_size=10)
    activeRFF1s.append(active_RF.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True))

pickle.dump(randRFF1s, open("randRFF1s.p", "wb" ))

pickle.dump(activeRFF1s, open("activeRFF1s.p", "wb" ))

print(datetime.datetime.now())
