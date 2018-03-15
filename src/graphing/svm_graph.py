import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

import pickle
import matplotlib.pyplot as plt


svmActiveF1 = {}
svmRandomF1 = {}
svmSysF1 ={}

# Graphing
plt.xlabel('Sample Size')
plt.ylabel('F1 Score')

for sampleSize in range(50, 201, 10):
    aSVMfname = "../tests/svm_results/activeSVMF1_"+str(sampleSize)+".p"
    rSVMfname = "../tests/svm_results/randSVMF1_"+str(sampleSize)+".p"
    sSVMfname = "../tests/systematic/svm_results/sysSVMF1_"+str(sampleSize)+".p"

    svmActiveF1[sampleSize] = pickle.load(open(aSVMfname, "rb"))
    svmRandomF1[sampleSize] = pickle.load(open(rSVMfname, "rb"))
    svmSysF1[sampleSize] = pickle.load(open(sSVMfname, "rb"))

    X = [sampleSize] * 150
    plt.scatter(X, svmRandomF1[sampleSize], s=5, c='Blue', alpha=0.05)
    plt.scatter([x + 1 for x in X], svmSysF1[sampleSize], s=5, c='purple', alpha=0.05)
    plt.scatter([x + 2 for x in X], svmActiveF1[sampleSize][10], s=5, c='Green', alpha=0.075)

# for sampleSize in range(210, 370, 10):
#     sSVMfname = "../tests/systematic/svm_results/sysSVMF1_"+str(sampleSize)+".p"
#     svmSysF1[sampleSize] = pickle.load(open(sSVMfname, "rb"))
#
#     X = [sampleSize] * 150
#
#     plt.scatter([x + 1 for x in X], svmSysF1[sampleSize], s=5, c='purple', alpha=0.05)

plt.show()
