from tester import Tester
import pickle
import sys, os
src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

rand_tester = pickle.load(open("results-rand0.p", "rb"))
active_tester0 = pickle.load(open("testers/tester_rand+act_16-25.p", "rb"))
active_tester1 = pickle.load(open("testers/tester_rand+act_26-35.p", "rb"))

active_results = {}
for size in active_tester0.currentResults:
    active_results[size] = [active_tester0.currentResults[size][0]]
for size in active_tester1.currentResults:
    active_results[size] = [active_tester1.currentResults[size][0]]
for size in rand_tester.currentResults:
    active_results[size].append(rand_tester.currentResults[size][0])

newTesterGrapher = Tester(['Active Net', 'Random Net'])
newTesterGrapher.currentResults = active_results
newTesterGrapher.graphResults()
