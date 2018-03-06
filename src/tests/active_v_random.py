import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

from data.dataset import Dataset
from models.model import Model
from models.activelearn import Active_Learner


data = Dataset('SUSY_100k.csv').random_sample(.01) #1k points
(total_train, total_test) = data.test_train_split(train_percent=.8)

rand_SVM800 = Model('SVM')
rand_SVM800.fit(total_train.get_x(), total_train.get_y())

train160 = total_train.random_sample(.2)
rand_SVM160 = Model('SVM')
rand_SVM160.fit(train160.get_x(), train160.get_y())

active_SVM = Model('SVM', active=True)
AL_SVM = Active_Learner(model=active_SVM, start_size=.1, end_size=.2, step_size=.005)
active_SVM = AL_SVM.fit(total_train.get_x(), total_train.get_y())

rand_SVM800.test(total_test.get_x(), total_test.get_y())

rand_SVM160.test(total_test.get_x(), total_test.get_y())

active_SVM.test(total_test.get_x(), total_test.get_y())
