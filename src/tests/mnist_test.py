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

mnist_sample = mnist_data.random_sample(percent=.1)
(train, test) = mnist_sample.test_train_split(train_percent=.6)
print(test.shape)
rand_train = train.random_sample(percent=.1)

mnist_pca = mnist_sample.pca(n_components=50)
(train_pca, test_pca) = mnist_pca.test_train_split(train_percent=.6)
rand_train_PCA = train_pca.random_sample(percent=.8)

# Delete old results
if os.path.isfile('results.txt'):
    os.remove('results.txt')

rand_SVM = Model('SVM')
rand_SVM.fit(rand_train.get_x(), rand_train.get_y())
rand_SVM.test(test.get_x(), test.get_y(), fname='results.txt')

active_SVM = Model('SVM', sample='Active')
AL_SVM = Active_Learner(model=active_SVM, start_size=.04, end_size=.08, step_size=.01)
active_SVM = AL_SVM.fit(train.get_x(), train.get_y())
active_SVM.test(test.get_x(), test.get_y(), fname='results.txt')
