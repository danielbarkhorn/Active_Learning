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

for sampleSize in range(240, 261, 20):
    startSize = int(((.5 * sampleSize) // 5) * 5)

    activeSVMF1s = {10:[]}
    aSVMF1fname = "svm_results/activeSVMF1_"+str(sampleSize)+".p"

    # run with this sample size this many times
    for _ in range(50):
        #getting test data to use for both models
        (train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.8)

        active_SVM = Model('SVM', sample='Active')
        active_SVM.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=startSize, end_size=sampleSize, step_size=10)
        activeSVMF1s[10].append(active_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True, avg='weighted'))

    pickle.dump(activeSVMF1s, open(aSVMF1fname, "wb" ))
