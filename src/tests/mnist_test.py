import os
import sys
import pickle
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

from data.dataset import Dataset
from models.model import Model
from models.activelearn import Active_Learner

mnist_data = pickle.load(open( "../data/pickled/mnist_data.p", "rb" ))

mnist_sample = mnist_data.random_sample(percent=.1) #4.2k instances
#(train, test) = mnist_sample.test_train_split(train_percent=.6)
#rand_train = train.random_sample(percent=.1)

mnist_pca = mnist_sample.pca(n_components=50)
(train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.6)
rand_train_PCA = train_pca.random_sample(size=250)

# Delete old results
if os.path.isfile('results.txt'):
    os.remove('results.txt')

rand_SVM = Model('SVM')
rand_SVM.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
rand_SVM.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')

active_SVM = Model('SVM', sample='Active')
active_SVM.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=150, end_size=250, step_size=10)
active_SVM.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')

rand_RF = Model('RF')
rand_RF.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
rand_RF.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')

active_RF = Model('RF', sample='Active')
active_RF.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=150, end_size=250, step_size=10)
active_RF.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')

rand_LR = Model('LR')
rand_LR.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
rand_LR.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')

active_LR = Model('LR', sample='Active')
active_LR.activeLearn(train_pca.get_x(), train_pca.get_y(), start_size=150, end_size=250, step_size=10)
active_LR.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')
