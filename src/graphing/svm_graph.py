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

act_means = []
rand_means = []

for sampleSize in range(50, 171, 10):
    aSVMfname = "../tests/svm_results/activeSVMF1_"+str(sampleSize)+".p"
    rSVMfname = "../tests/svm_results/randSVMF1_"+str(sampleSize)+".p"

    svmActiveF1[sampleSize] = pickle.load(open(aSVMfname, "rb"))
    svmRandomF1[sampleSize] = pickle.load(open(rSVMfname, "rb"))

    # Graphing
    X = [sampleSize] * 150
    plt.scatter(X, svmRandomF1[sampleSize], s=5, c='Blue', alpha=0.05)
    plt.scatter([x + 1 for x in X], svmActiveF1[sampleSize][10], s=5, c='Red', alpha=0.075)

    rand_means.append(np.mean(svmRandomF1[sampleSize]))
    act_means.append(np.mean(svmActiveF1[sampleSize][10]))

plt.plot(range(50, 171, 10), rand_means)
plt.plot(range(50, 171, 10), act_means)
plt.show()
