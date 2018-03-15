import pickle
import numpy as np

for sampleSize in range(50, 161, 10):

    aSVMfname = "svm_results/activeSVMF1_"+str(sampleSize)+".p"
    rSVMfname = "svm_results/randSVMF1_"+str(sampleSize)+".p"

    aRFfname = "rf_results/activeRFF1_"+str(sampleSize)+".p"
    rRFfname = "rf_results/randRFF1_"+str(sampleSize)+".p"

    resultsSVMActive = pickle.load(open( aSVMfname, "rb" ))
    print('Active SVM', sampleSize)
    for step in resultsSVMActive:
        print(step)
        print(np.min(resultsSVMActive[step]), np.mean(resultsSVMActive[step]), np.max(resultsSVMActive[step]))

    resultsSVMRandom = pickle.load(open( rSVMfname, "rb" ))
    print('\n', 'Random SVM', sampleSize)
    print(np.min(resultsSVMRandom), np.mean(resultsSVMRandom), np.max(resultsSVMRandom), '\n')

    resultsRFActive = pickle.load(open( aRFfname, "rb" ))
    print('Active RF', sampleSize)
    for step in resultsRFActive:
        print(step)
        print(np.min(resultsRFActive[step]), np.mean(resultsRFActive[step]), np.max(resultsRFActive[step]))


    resultsRFRandom = pickle.load(open( rRFfname, "rb" ))
    print('\n', 'Random RF', sampleSize)
    print(np.min(resultsRFRandom), np.mean(resultsRFRandom), np.max(resultsRFRandom), '\n')

    print()
