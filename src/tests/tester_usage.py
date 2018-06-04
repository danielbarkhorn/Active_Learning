from tester import Tester
from data.dataset import Dataset
from models.model import Model
import pickle

mnist_pca = pickle.load(open( "../data/pickled/mnist_data_pca50.p", "rb" ))

mnist_pca_sample = mnist_pca.random_sample(percent=.2) #~10k samples

models = [Model('SVM', sample='Active', name='Active SVM'), Model('SVM', name='Random SVM')]

myTester = Tester(models)

<<<<<<< Updated upstream
sizes = [(100, 20, 200),
         (120, 20, 220),
         (140, 20, 240),
         (160, 20, 260),
         (180, 20, 280)]

results = myTester.runTests(mnist_pca_sample, sizes, iterations = 8)

pickle.dump(myTester, open("tester.p", "wb" ))
=======
sizes = [(1400, 20, 1600),
         (1500, 20, 1700),
         (1600, 20, 1800),
         (1700, 20, 1900),
         (1800, 20, 2000),
         (1900, 20, 2100),
         (2000, 20, 2200),
         (2100, 20, 2300),
         (2200, 20, 2400),
         (2300, 20, 2500),]

results = myTester.runTests(mnist_sample, sizes, iterations = 150)
pickle.dump(myTester, open("testers/tester_rand+act_16-25.p", "wb" ))
#myTester.graphResults()
>>>>>>> Stashed changes
