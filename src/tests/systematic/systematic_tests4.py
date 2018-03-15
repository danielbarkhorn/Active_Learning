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

mnist_pca_sample = mnist_pca.random_sample(percent=.5) #24 instances

for sampleSize in range(300, 401, 10):
    print(sampleSize)
    sysSVMF1s = []
    sysSVMF1fname = "svm_results/sysSVMF1_"+str(sampleSize)+".p"

    for _ in range(150):
        #getting test data to use for models
        (train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.8)

        #make random train data and model
        sys_train_PCA = train_pca.systematic_sample(size=sampleSize, sort='magnitude')

        sys_SVM = Model('SVM')
        sys_SVM.fit(sys_train_PCA.get_x(), sys_train_PCA.get_y())

        sysSVMF1s.append(sys_SVM.test_metric(test_pca.get_x(), test_pca.get_y(), f1=True))

    pickle.dump(sysSVMF1s, open(sysSVMF1fname, "wb" ))

print(datetime.datetime.now())
