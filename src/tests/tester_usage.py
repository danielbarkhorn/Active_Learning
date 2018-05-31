from tester import Tester
from data.dataset import Dataset
from models.model import Model
import pickle

mnist_pca = pickle.load(open( "../data/pickled/mnist_data_pca50.p", "rb" ))

mnist_pca_sample = mnist_pca.random_sample(percent=.2) #~10k samples

models = [Model('SVM', sample='Active', name='Active SVM'), Model('SVM', name='Random SVM')]

myTester = Tester(models)

sizes = [(100, 20, 200),
         (120, 20, 220),
         (140, 20, 240),
         (160, 20, 260),
         (180, 20, 280)]

results = myTester.runTests(mnist_pca_sample, sizes, iterations = 8)

pickle.dump(myTester, open("tester.p", "wb" ))
