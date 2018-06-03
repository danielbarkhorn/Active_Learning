import os
import sys
import pickle
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)
from data.dataset import Dataset
from models.model import Model


mnist = pickle.load(open( "data/pickled/mnist_data.p", "rb" ))

mnist_sample = mnist.random_sample(percent=.5) #24 instances

(train, test) = mnist.test_train_split(train_percent=.8)

NN = Model('NN', name='testNN')

train_500 = train.random_sample(size=500)
train_500_x = train_500.get_x() / 256
train_500_y = train_500.get_y()

NN.fit(train_500_x, train_500_y)
