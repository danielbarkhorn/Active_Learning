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

#for sampleSize in range(100, 551, 15):
for sampleSize in range(250, 251, 10):
    startSize = int(((.5 * sampleSize) // 5) * 5)

    randSVMF1s = []
    activeSVMF1s = {5:[], 10:[], 15:[]}

    rfname = "svm_results/randSVMF1_"+str(sampleSize)+".p"
    afname = "svm_results/activeSVMF1_"+str(sampleSize)+".p"

    for _ in range(2):
        #getting test data to use for both models
        (train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.8)

        #make random train data and model
        rand_train_PCA = train_pca.random_sample(size=sampleSize)
        rand_SVM = Model('SVM')
        rand_SVM.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
        randSVMF1s.append(rand_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True, avg='weighted'))

        #make active model for step size 5, 10, 15
        for stepSize in [5,10,15]:
            active_SVM = Model('SVM', sample='Active')
            active_SVM.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=startSize, end_size=sampleSize, step_size=stepSize)
            activeSVMF1s[stepSize].append(active_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True, avg='weighted'))

    pickle.dump(randSVMF1s, open(rfname, "wb" ))

    pickle.dump(activeSVMF1s, open(afname, "wb" ))

print(datetime.datetime.now())
