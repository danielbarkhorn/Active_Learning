import pickle
import numpy as np

for sampleSize in range(210, 211, 1):

    rfname = "svm_results/randSVMF1_"+str(sampleSize)+".p"
    afname = "svm_results/activeSVMF1_"+str(sampleSize)+".p"

    resultsActive = pickle.load(open( rfname, "rb" ))
    print('Active', sampleSize)
    print(np.min(resultsActive), np.mean(resultsActive), np.max(resultsActive), '\n')

    resultsRandom = pickle.load(open( afname, "rb" ))
    print('Random', sampleSize)
    print(np.min(resultsRandom), np.mean(resultsRandom), np.max(resultsRandom), '\n')
