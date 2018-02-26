from numpy import zeros, ndarray, linalg, append
from random import random

class Sampling:
    # data as numpy ndarray with structure
    # [[y_0, x_00, x_01, ... , x_0n],
    #               ...
    #  [y_n, x_n0, x_n1, ... , x_nn]]
    def __init__(self, data, data_y=None):
            if data_y:
                self.data = zeros(data.shape[0], data.shape[1]+1)
                for ind in range(self.data_y.shape[0]):
                    self.data = append(data_y[ind], data[ind])
            self.data = data
            self.shape = data.shape

    def systematic(self, percent=0.25, sort='feature'):
        sample_shape = (int(self.shape[0]*percent), self.shape[1])
        sample = zeros(sample_shape)
        step = int((1/percent))
        k = int(random() * (1/percent))

        if sort == 'magnitude':
            magnitudes_order = zeros(self.shape[0])
            magnitudes_data = zeros(self.shape)
            for ind in range(self.shape[0]):
                magnitudes_order[ind] = linalg.norm(self.data[ind][1:])
            magnitudes_order = magnitudes_order.argsort()
            magnitudes_data = self.data[magnitudes_order]

            for ind in range(sample_shape[0]):
                sample[ind] = magnitudes_data[k]
                k += step

            return sample

        return
