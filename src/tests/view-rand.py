import os
import sys
import pickle
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)
import models.model
import tests.tester

tester = pickle.load(open('testers/tester_rand+act_26-35.p', 'rb'))

tester.graphResults()
