from tester import Tester
import pickle
import sys,os
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

grapherTester = pickle.load(open("finalGraphTester.p", "rb"))

grapherTester.graphResults()
