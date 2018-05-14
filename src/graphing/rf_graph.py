import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

#TODO:
# Make this a class such that we can create graphs more easily

import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

rfActiveF1 = {}
rfRandomF1 = {}
svmSysF1 = {}
act_means = []
rand_means = []
sys_means = []


# Graphing
plt.xlabel('Sample Size')
plt.ylabel('F1 Score')

for sampleSize in range(50, 211, 10):
    aRFfname = "../tests/rf_results/activeRFF1_"+str(sampleSize)+".p"
    rRFfname = "../tests/rf_results/randRFF1_"+str(sampleSize)+".p"
    sSVMfname = "../data/ActiveLearning_data/systematic_svm_results/sysSVMF1_"+str(sampleSize)+".p"

    rfActiveF1[sampleSize] = pickle.load(open(aRFfname, "rb"))
    rfRandomF1[sampleSize] = pickle.load(open(rRFfname, "rb"))
    svmSysF1[sampleSize] = pickle.load(open(sSVMfname, "rb"))

    X = [sampleSize] * 150
    plt.scatter(X, rfRandomF1[sampleSize], s=8, c='Blue', alpha=0.075)
    plt.scatter(X, rfActiveF1[sampleSize][10], s=8, c='Red', alpha=0.075)
    #plt.scatter(X, svmSysF1[sampleSize][:50], s=8, c='Green', alpha=0.075)

    rand_means.append(np.mean(rfRandomF1[sampleSize]))
    act_means.append(np.mean(rfActiveF1[sampleSize][10]))
    sys_means.append(np.mean(svmSysF1[sampleSize][10]))

plt.plot(range(50, 211, 10), rand_means,linewidth=3.0,alpha=0.5)
plt.plot(range(50, 211, 10), act_means,linewidth=3.0, c='Red',alpha=0.5)

red_patch = mpatches.Patch(color='red', label='Active')
blu_patch = mpatches.Patch(color='blue', label='Random')
plt.legend(handles=[red_patch,blu_patch],loc=(0.75, 0.05))
#plt.plot(range(50, 301, 10), sys_means,linewidth=3.0, c='Green',alpha=0.5)
plt.show()
