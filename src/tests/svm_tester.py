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

models = [('SVM', 'Active', 'Active-SVM-boosted')]

myTester = Tester(models)

sizes = [(end//2, 10, end, 2, 1) for end in range(50, 310, 10)]

results = myTester.runTests(mnist_sample, sizes, iterations = 150)
pickle.dump(myTester, open("testers/tester_SVM-act_16-25.p", "wb" ))
#myTester.graphResults()
