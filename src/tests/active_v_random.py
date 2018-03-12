import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

from data.dataset import Dataset
from models.model import Model
from models.activelearn import Active_Learner

# Make our data
data = Dataset('SUSY_100k.csv').random_sample(.01) #1k points
data = data.pca(n_components=5)
(total_train, total_test) = data.test_train_split(train_percent=.8)
train160 = total_train.random_sample(.05)
sys_train160 = total_train.systematic_sample(percent=0.05)

# Make our models
rand_SVM800 = Model('SVM')
rand_SVM800.fit(total_train.get_x(), total_train.get_y())

rand_SVM160 = Model('SVM')
rand_SVM160.fit(train160.get_x(), train160.get_y())

sys_SVM160 = Model('SVM', sample='Systematic')
sys_SVM160.fit(sys_train160.get_x(), sys_train160.get_y())

active_SVM = Model('SVM', sample='Active')
AL_SVM = Active_Learner(model=active_SVM, start_size=.01, end_size=.05, step_size=.005)
active_SVM = AL_SVM.fit(total_train.get_x(), total_train.get_y())

# Delete old results
if os.path.isfile('results.txt'):
    os.remove('results.txt')

# Test our models
rand_SVM800.test(total_test.get_x(), total_test.get_y(), fname='results.txt')

rand_SVM160.test(total_test.get_x(), total_test.get_y(), fname='results.txt')

sys_SVM160.test(total_test.get_x(), total_test.get_y(), fname='results.txt')

active_SVM.test(total_test.get_x(), total_test.get_y(), fname='results.txt')
