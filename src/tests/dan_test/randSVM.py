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

mnist_pca = pickle.load(open( "../../data/pickled/mnist_data_pca50.p", "rb" ))

for sampleSize in range(220, 301, 10):

    rSVMF1fname = "rand_svm_results/randSVMF1_"+str(sampleSize)+".p"
    randSVMF1s = []

    # run with this sample size this many times
    for _ in range(50):
        #getting test data to use for both models
        (train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.8)

        #make random train data and models
        rand_train_PCA = train_pca.random_sample(size=sampleSize)

        rand_SVM = Model('SVM')
        rand_SVM.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
        randSVMF1s.append(rand_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True, avg='weighted'))

    pickle.dump(randSVMF1s, open(rSVMF1fname, "wb" ))

print(datetime.datetime.now())
