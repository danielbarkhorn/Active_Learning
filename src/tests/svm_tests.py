import datetime
import os
import sys
import pickle
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

import numpy as np
from tqdm import tqdm

from data.dataset import Dataset
from models.model import Model

print(datetime.datetime.now())


mnist_pca = pickle.load(open( "../data/pickled/mnist_data_pca50.p", "rb" ))

mnist_pca_sample = mnist_pca.random_sample(percent=.5) #24 instances

#randSVMF1s = []
activeSVMF1s = []

for _ in range(5):
    #getting test data to use for both models
    (train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.8)

    #make active model
    active_SVM = Model('SVM', sample='Active')
    active_SVM.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=10, end_size=150, step_size=10, random_size=2, outlier_size=1)
    activeSVMF1s.append(active_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True))

#pickle.dump(randSVMF1s, open("randSVMF1s.p", "wb" ))

pickle.dump(activeSVMF1s, open("activeSVMDistF1s150.p", "wb" ))

print(datetime.datetime.now())

print(activeSVMF1s)
