import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

import pickle
import matplotlib.pyplot as plt


svmActiveF1 = {}
svmRandomF1 = {}

for sampleSize in range(50, 171, 10):
    aSVMfname = "../tests/svm_results/activeSVMF1_"+str(sampleSize)+".p"
    rSVMfname = "../tests/svm_results/randSVMF1_"+str(sampleSize)+".p"

    svmActiveF1[sampleSize] = pickle.load(open(aSVMfname, "rb"))
    svmRandomF1[sampleSize] = pickle.load(open(rSVMfname, "rb"))

    # Graphing
    X = [sampleSize] * 150
    plt.scatter(X, svmRandomF1[sampleSize], s=5, c='Blue', alpha=0.05)

    plt.scatter([x + 1 for x in X], svmActiveF1[sampleSize][5], s=5, c='Red', alpha=0.075)
    plt.scatter([x + 2 for x in X], svmActiveF1[sampleSize][10], s=5, c='Green', alpha=0.075)
    plt.scatter([x + 3 for x in X], svmActiveF1[sampleSize][15], s=5, c='Orange', alpha=0.075)

plt.show()
