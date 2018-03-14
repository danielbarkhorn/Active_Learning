import pickle
import numpy as np

for sampleSize in range(100, 101, 1):

    aSVMfname = "svm_results/activeSVMF1_"+str(sampleSize)+".p"
    rSVMfname = "svm_results/randSVMF1_"+str(sampleSize)+".p"

    aRFfname = "rf_results/activeRFF1_"+str(sampleSize)+".p"
    rRFfname = "rf_results/randRFF1_"+str(sampleSize)+".p"

    resultsSVMActive = pickle.load(open( aSVMfname, "rb" ))
    print('Active SVM', sampleSize)
    print(resultsSVMActive)
    #print(np.min(resultsActive), np.mean(resultsActive), np.max(resultsActive), '\n')

    resultsSVMRandom = pickle.load(open( rSVMfname, "rb" ))
    print('Random SVM', sampleSize)
    print(np.min(resultsSVMRandom), np.mean(resultsSVMRandom), np.max(resultsSVMRandom), '\n')

    resultsRFActive = pickle.load(open( aRFfname, "rb" ))
    print('Active RF', sampleSize)
    print(resultsRFActive)

    resultsRFRandom = pickle.load(open( rRFfname, "rb" ))
    print('Random RF', sampleSize)
    print(np.min(resultsRFRandom), np.mean(resultsRFRandom), np.max(resultsRFRandom), '\n')
