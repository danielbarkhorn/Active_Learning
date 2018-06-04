import datetime
import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm
<<<<<<< Updated upstream
=======
from models.model import Model
import glob
import gc
>>>>>>> Stashed changes

src_path = os.getcwd().split('/')
src_path = '/'.join(src_path[:src_path.index('src')+1])
sys.path.append(src_path)

class Tester(object):
    def __init__(self, models):
        self.models = models
        self.currentResults = None

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
        for size in tqdm(sizes):
            size_F1s = [[] for _ in self.models]
<<<<<<< Updated upstream
            for iteration in range(iterations):
                (train, test) = data.test_train_split(train_percent=.8)
=======
            for iteration in tqdm(range(iterations)):
                (train, test) = data.test_train_split(train_percent=.9)
>>>>>>> Stashed changes

                r_size = 0 if len(size) < 4 else size[3]
                o_size = 0 if len(size) < 4 else size[4]

                # train and test models
<<<<<<< Updated upstream
                for i in tqdm(range(len(self.models))):
                    model = self.models[i]
=======
                for i in range(len(self.models)):
                    model = Model(type=self.models[i][0], sample=self.models[i][1], name=(self.models[i][2]+str(iteration)))
>>>>>>> Stashed changes
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

        self.currentResults = results
        return results

    def graphResults(self):
        results = self.currentResults
        print(results)
        plt.xlabel('Sample Size')
        plt.ylabel('F1 Score')
        modelColors = []
        for model in self.models:
            modelColors.append(np.random.rand(3,))
        for size in results:
            X = [size]*len(results[size][0])
            for model_ind in range(len(self.models)):
                plt.scatter(X, results[size][model_ind], s=8, c=modelColors[model_ind], alpha=0.05)
        patches = []
        for i in range(len(self.models)):
            patches.append(mpatches.Patch(color=modelColors[i], label=self.models[i].name))
        plt.legend(handles=patches,loc=(0.75, 0.05))
        plt.show()
        return
