import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

from data.dataset import Dataset
from models.model import Model
from models.activelearn import Active_Learner


data = Dataset('SUSY_100k.csv').random_sample(.01) #1k points
(total_train, total_test) = data.test_train_split(train_percent=.7)

rand_SVM = Model('SVM')
rand_SVM.fit(total_train.get_x(), total_train.get_y())

active_SVM = Model('SVM')
