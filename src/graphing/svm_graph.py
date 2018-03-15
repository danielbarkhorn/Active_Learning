import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

import pickle
import matplotlib.pyplot as plt
import numpy as np

svmActiveF1 = {}
svmRandomF1 = {}
svmSysF1 = {}
act_means = []
rand_means = []


# Graphing
plt.xlabel('Sample Size')
plt.ylabel('F1 Score')

for sampleSize in range(50, 301, 10):
    aSVMfname = "../data/ActiveLearning_data/active_svm_results/activeSVMF1_"+str(sampleSize)+".p"
    rSVMfname = "../data/ActiveLearning_data/rand_svm_results/randSVMF1_"+str(sampleSize)+".p"
    sSVMfname = "../data/ActiveLearning_data/systematic_svm_results/sysSVMF1_"+str(sampleSize)+".p"

    svmActiveF1[sampleSize] = pickle.load(open(aSVMfname, "rb"))
    svmRandomF1[sampleSize] = pickle.load(open(rSVMfname, "rb"))
    svmSysF1[sampleSize] = pickle.load(open(sSVMfname, "rb"))

    X = [sampleSize] * 150
    plt.scatter([x - 0.5 for x in X], svmRandomF1[sampleSize], s=5, c='Blue', alpha=0.05)
    plt.scatter([x + 0.5 for x in X], svmActiveF1[sampleSize][10], s=5, c='Red', alpha=0.075)

    rand_means.append(np.mean(svmRandomF1[sampleSize]))
    act_means.append(np.mean(svmActiveF1[sampleSize][10]))

plt.plot(range(50, 301, 10), rand_means,linewidth=3.0,alpha=0.5)
plt.plot(range(50, 301, 10), act_means,linewidth=3.0, c='Red',alpha=0.5)
plt.show()
