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
(train, test) = mnist_sample.test_train_split(train_percent=.6)
rand_train = train.random_sample(percent=.1)

mnist_pca = mnist_sample.pca(n_components=50)
(train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.6)
rand_train_PCA = train_pca.random_sample(percent=.08)

# Delete old results
if os.path.isfile('results.txt'):
    os.remove('results.txt')

rand_SVM = Model('SVM')
rand_SVM.fit(rand_train_PCA.get_x(), rand_train_PCA.get_y())
rand_SVM.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')

active_SVM = Model('SVM', sample='Active')
AL_SVM = Active_Learner(model=active_SVM, start_size=100, end_size=250, step_size=10)
active_SVM = AL_SVM.fit(train_pca.get_x(), train_pca.get_y())
active_SVM.test(test_pca.get_x(), test_pca.get_y(), fname='results.txt')
