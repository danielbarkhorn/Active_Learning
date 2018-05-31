import pickle
import tester

myTester = pickle.load(open( "tester.p", "rb" ))

myTester.graphResults()
