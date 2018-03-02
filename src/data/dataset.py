import numpy as np
import random
import os

class Dataset:
    def __init__(self, filename="SUSY_sample.csv", data=None, data_y=None):
        if(data_y):
            assert(data is None), 'data_y given without data_x'
            self.data = np.zeros(data.shape[0], data.shape[1]+1)
            for ind in range(self.data_y.shape[0]):
                self.data = np.append(data_y[ind], data[ind])
            self.shape = self.data.shape
        elif data is not None:
            self.data = data
            self.shape = data.shape
        else:
            path_arr = os.getcwd().split('/')
            filepath = '..'*(len(path_arr)-1-path_arr.index('src'))
            filepath += "/data/generated/"+filename
            self.data = np.loadtxt(filepath, delimiter=',')
            self.shape = self.data.shape

    def get_x(self):
        return self.data[:,1:]

    def get_y(self):
        return self.data[:,0]

    def random_sample(self, percent=0.25):
        sample_shape = (int(self.shape[0]*percent), self.shape[1])
        rand_ind = random.sample(range(self.shape[0]), sample_shape[0])
        rand_data = self.data[rand_ind]
        rand_dataset = Dataset(data=rand_data)
        return rand_dataset

    def systematic_sample(self, percent=0.25, sort='feature'):
        sample_shape = (int(self.shape[0]*percent), self.shape[1])
        sample = np.zeros(sample_shape)

        if sort == 'magnitude':
            step = int((1/percent))
            k = int(random.random() * (1/percent))
            magnitudes_order = np.zeros(self.shape[0])
            magnitudes_data = np.zeros(self.shape)
            for ind in range(self.shape[0]):
                magnitudes_order[ind] = np.linalg.norm(self.data[ind][1:])
            magnitudes_order = magnitudes_order.argsort()
            magnitudes_data = self.data[magnitudes_order]

            for ind in range(sample_shape[0]):
                sample[ind] = magnitudes_data[k]
                k += step

            sample_dataset = Dataset(data=sample)
            return sample_dataset
        else:
            window = int(sample_shape[0]/sample_shape[1])
            step = int((self.shape[0]*percent/self.shape[1]))
            att_order = np.zeros(self.shape[0])

            for att in range(self.shape[1]):
                k = int(random.random() * window)
                att_order = self.data[:, att].argsort()
                att_data = self.data[att_order]

                for ind in range(window):
                    sample[ind + (att * window)] = att_data[k]
                    k += step

            sample_dataset = Dataset(data=sample)
            return sample_dataset

    def save(self, filename):
        np.savetxt(filename, self.data, delimiter=',')
