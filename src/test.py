from data import dataset as d
import numpy as np

test_data = np.random.randn(100, 5) + 1
test_sample = s.Sampler(test_data)

susy_data = d.Dataset()

print(test_sample.systematic(sort='feature'))
