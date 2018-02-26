import numpy as np
from random import random as rand

class Sampling:
    # data as numpy ndarray with structure
    # [[y_0, x_00, x_01, ... , x_0n],
    #               ...
    #  [y_n, x_n0, x_n1, ... , x_nn]]
    def __init__(self, data, data_y=None):
            if data_y:
                self.data = np.zeros(data.shape[0], data.shape[1]+1)
                for ind in range(self.data_y.shape[0]):
                    self.data = np.append(data_y[ind], data[ind])
            self.data = data
            self.shape = data.shape

    def systematic(self, percent=0.25, sort='feature'):
        sample_shape = (int(self.shape[0]*percent), self.shape[1])
        sample = np.zeros(sample_shape)
        step = int((1/percent))
        k = int(rand() * (1/percent))

        if sort == 'magnitude':
            magnitudes_order = np.zeros(self.shape[0])
            magnitudes_data = np.zeros(self.shape)
            for ind in range(self.shape[0]):
                magnitudes_order[ind] = np.linalg.norm(self.data[ind][1:])
            magnitudes_order = magnitudes_order.argsort()
            magnitudes_data = self.data[magnitudes_order]

            for ind in range(sample_shape[0]):
                sample[ind] = magnitudes_data[k]
                k += step

            return sample
        else:
            return
            #TODO:
            # Implement sorting by features, then choosing points with systematic apprch
            # Question, do we care about datapoints being included repeatedly?
        return

    def random(self, percent=0.25):
        sample_shape = (int(self.shape[0]*percent), self.shape[1])
        rand_ind = np.random.randint(0, high=self.shape[0], size=sample_shape[0])
        return self.data[rand_ind]
