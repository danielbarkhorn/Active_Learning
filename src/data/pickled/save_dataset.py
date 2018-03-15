import pickle
import os
import sys
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

from data.dataset import Dataset

mnist_data = Dataset(filename='MNIST.csv', header=1)
#pickle.dump(mnist_data, open("mnist_data.p", "wb" ))

mnist_pca = mnist_data.pca(n_components=50)
pickle.dump(mnist_pca, open("mnist_data_pca50.p", "wb" ))

# susy_data = Dataset('SUSY_100k.csv')
# pickle.dump(susy_data, open("susy_data.p", "wb"))
