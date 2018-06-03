import os
import sys
import pickle
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)
from tester import Tester
from data.dataset import Dataset
from models.model import Model
import pickle

mnist = pickle.load(open( "../data/pickled/mnist_data.p", "rb" ))

mnist_sample = mnist.random_sample(percent=.5) #24 instances

models = [Model(type='NN', sample='Active', name='Active-NN')]

myTester = Tester(models)

sizes = [(1300, 20, 1500),
         (1400, 20, 1600),
         (1500, 20, 1700),
         (1600, 20, 1800),
         (1700, 20, 1900),
         (1800, 20, 2000),
         (1900, 20, 2100)]

results = myTester.runTests(mnist_sample, sizes, iterations = 50)
pickle.dump(results, open("results.p", "wb" ))
