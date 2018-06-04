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

models = [('NN', 'Random', 'Random-NN')]

myTester = Tester(models)

sizes = [(1400, 20, 1600),
         (1500, 20, 1700),
         (1600, 20, 1800),
         (1700, 20, 1900),
         (1800, 20, 2000),
         (1900, 20, 2100),
         (2000, 20, 2200),
         (2100, 20, 2300),
         (2200, 20, 2400),
         (2300, 20, 2500),
         (2400, 20, 2600),
         (2500, 20, 2700),
         (2600, 20, 2800),
         (2700, 20, 2900),
         (2800, 20, 3000),
         (2900, 20, 3100),
         (3000, 20, 3200),
         (3100, 20, 3300),
         (3200, 20, 3400),
         (3300, 20, 3500),]

results = myTester.runTests(mnist_sample, sizes, iterations = 50)
pickle.dump(myTester, open("testers/tester_rand+act_16-35.p", "wb" ))
myTester.graphResults()
