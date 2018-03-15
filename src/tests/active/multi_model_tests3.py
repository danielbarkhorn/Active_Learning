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

for sampleSize in range(220, 300, 10):
    startSize = int(((.5 * sampleSize) // 5) * 5)

    TESTDATA = []
    RANDTRAIN = []

    randSVMF1s = []
    randSVMRaw = []
    activeSVMF1s = {5:[], 10:[], 15:[]}
    activeSVMRaw = {5:[], 10:[], 15:[]}

    testfnam = "test_data/test_data_"+str(sampleSize)+".p"
    randtrainfnam = "rand_train_data/rand_train_data_"+str(sampleSize)+".p"

    rSVMF1fname = "svm_results/randSVMF1_"+str(sampleSize)+".p"
    rSVMRawfname = "svm_results/randSVMRaw_"+str(sampleSize)+".p"
    aSVMF1fname = "svm_results/activeSVMF1_"+str(sampleSize)+".p"
    aSVMRawfname = "svm_results/activeSVMRaw_"+str(sampleSize)+".p"

    # run with this sample size this many times
    for _ in range(150):
        #getting test data to use for both models
        (train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.8)

        #make random train data and models
        rand_train_PCA = train_pca.random_sample(size=sampleSize)

        TESTDATA.append(test_pca)
        RANDTRAIN.append(rand_train_PCA)

        rand_SVM = Model('SVM')
        rand_SVM.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
        randSVMF1s.append(rand_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True, avg='weighted'))
        randSVMRaw.append(rand_SVM.predict(test_pca.get_x()))

        #make active model for step size 5, 10, 15
        for stepSize in [5,10,15]:
            active_SVM = Model('SVM', sample='Active')
            active_SVM.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=startSize, end_size=sampleSize, step_size=stepSize)
            activeSVMF1s[stepSize].append(active_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True, avg='weighted'))
            activeSVMRaw[stepSize].append(active_SVM.predict(test_pca.get_x()))

    pickle.dump(TESTDATA, open(testfnam, "wb"))
    pickle.dump(RANDTRAIN, open(randtrainfnam, "wb"))

    pickle.dump(randSVMF1s, open(rSVMF1fname, "wb" ))
    pickle.dump(randSVMRaw, open(rSVMRawfname, "wb"))
    pickle.dump(activeSVMF1s, open(aSVMF1fname, "wb" ))
    pickle.dump(activeSVMRaw, open(aSVMRawfname, "wb"))

print(datetime.datetime.now())
