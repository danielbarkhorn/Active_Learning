import datetime
import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

class Tester(object):
    def __init__(self, models):
        self.models = models

    def runTests(self, data, sizes, iterations, graph=False):
        """
        Parameters
        ----------
        data : dataset
        data to be sampled from for testing

        sizes : array
        array containing tuples of
            (start size,
            step size,
            end size,
            random_size (optional),
            outlier_size (optional))
        to be used for testing.

        iterations : int
        Number of times to run each test defined by sizes array
        """

        results = {}
        for size in sizes:
            size_F1s = [[] for _ in self.models]
            for iteration in range(iterations):
                (train, test) = data.test_train_split(train_percent=.8)

                r_size = 0 if len(size) < 4 else size[3]
                o_size = 0 if len(size) < 4 else size[4]

                # train and test models
                for i in tqdm(range(len(self.models))):
                    model = self.models[i]
                    if model.sample == 'Active':
                        model.activeLearn(train.get_x(),
                                          train.get_y(),
                                          start_size=size[0],
                                          end_size=size[2],
                                          step_size=size[1],
                                          random_size=r_size,
                                          outlier_size=o_size)
                    else:
                        rand_train = train.random_sample(size=size[2])
                        model.fit(rand_train.get_x(), rand_train.get_y())

                    size_F1s[i].append(model.test_metric(test.get_x(),
                                                         test.get_y(),
                                                         f1=True))
            results[size[2]] = size_F1s

        return results

    # def graphResults(self, results):
    #     plt.xlabel('Sample Size')
    #     plt.ylabel('F1 Score')
    #     for size in results:
    #         X = [size]*len(results[size][0])
    #         for model in results[size]:
    #             plt.scatter(X, model, s=8, alpha=0.075)
    #     plt.show()
    #     return
