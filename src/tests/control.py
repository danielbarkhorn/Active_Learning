import os
import sys
import pickle
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

from data.dataset import Dataset
from models.model import Model
from models.activelearn import Active_Learner

mnist_data = pickle.load(open( "../data/pickled/mnist_data.p", "rb" ))
