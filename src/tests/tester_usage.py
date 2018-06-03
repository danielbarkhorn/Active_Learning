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

models = [Model(type='NN', sample='Active', name='Active-NN'), Model(type='NN', name='Random-NN')]

myTester = Tester(models)

sizes = [(100, 20, 200)]

results = myTester.runTests(mnist_sample, sizes, iterations = 8)
print(results)
pickle.dump(myTester, open("tester.p", "wb" ))
