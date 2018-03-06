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

rand_SVM700 = Model('SVM')
rand_SVM700.fit(total_train.get_x(), total_train.get_y())

train160 = total_train.random_sample(.2)
rand_SVM160 = Model('SVM')
rand_SVM160.fit(train160.get_x(), train160.get_y())

active_SVM = Model('SVM')
AL_SVM = Active_Learner(model=active_SVM, start_size=.1, end_size=.2, step_size=.005)
active_SVM = AL_SVM.fit(total_train.get_x(), total_train.get_y())

from sklearn.metrics import classification_report

print('Random SVM, 700 pts')
print (classification_report(total_test.get_y(),rand_SVM700.predict(total_test.get_x(), proba=False)))

print('Random SVM, 160 pts')
print (classification_report(total_test.get_y(),rand_SVM160.predict(total_test.get_x(), proba=False)))

print('Active Learned SVM, 160 pts')
print (classification_report(total_test.get_y(),active_SVM.predict(total_test.get_x(), proba=False)))
