import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tests import tester

svmActiveF1 = {}
svmRandomF1 = {}
svmSysF1 = {}
act_means = []
rand_means = []
sys_means = []


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

    X = [sampleSize] * 50
    #plt.scatter(X, svmRandomF1[sampleSize][:50], s=8, c='Blue', alpha=0.075)
    plt.scatter(X, svmActiveF1[sampleSize][10][:50], s=8, c='Red', alpha=0.075)
    #plt.scatter(X, svmSysF1[sampleSize][:50], s=8, c='Green', alpha=0.075)

    rand_means.append(np.mean(svmRandomF1[sampleSize]))
    act_means.append(np.mean(svmActiveF1[sampleSize][10]))
    sys_means.append(np.mean(svmSysF1[sampleSize][10]))

#plt.plot(range(50, 301, 10), rand_means,linewidth=3.0,alpha=0.5)
plt.plot(range(50, 301, 10), act_means,linewidth=3.0, c='Red',alpha=0.5)


red_patch = mpatches.Patch(color='red', label='Active SVM Probabilistic')
#plt.plot(range(50, 301, 10), sys_means,linewidth=3.0, c='Green',alpha=0.5)
print(sys.path)
res = pickle.load(open('../tests/testers/tester_SVM-act_16-25.p', 'rb'))
res.graphResults()
