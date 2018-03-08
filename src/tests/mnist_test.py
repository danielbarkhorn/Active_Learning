import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

from data.dataset import Dataset
from models.model import Model
from models.activelearn import Active_Learner

mnist_data = Dataset(filename='MNIST.csv', header=1).random_sample(percent=.1)
mnist_pca = mnist_data.pca(n_components=50)
(train, test) = mnist_pca.test_train_split(train_percent=.8)
rand_data = train.random_sample(percent=.2)

rand_SVM = Model('KNN')
rand_SVM.fit(rand_data.get_x(), rand_data.get_y())
rand_SVM.test(test.get_x(), test.get_y(), fname='results.txt')

# active_SVM = Model('SVM', sample='Active')
# AL_SVM = Active_Learner(model=active_SVM, start_size=.1, end_size=.2, step_size=.05)
# active_SVM = AL_SVM.fit(train.get_x(), train.get_y())
# active_SVM.test(test.get_x(), test.get_y(), fname='results.txt')
