from tester import Tester
from data.dataset import Dataset
from models.model import Model
import pickle

mnist_pca = pickle.load(open( "../data/pickled/mnist_data_pca50.p", "rb" ))

mnist_pca_sample = mnist_pca.random_sample(percent=.2) #~10k samples

models = [Model('SVM', sample='Active'), Model('SVM')]

myTester = Tester(models)

sizes = [(100, 20, 200),
         (150, 20, 250, 3, 3)]

print(myTester.runTests(mnist_pca_sample, sizes, iterations = 8))
