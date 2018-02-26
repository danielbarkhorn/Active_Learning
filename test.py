from sampling import Sampling
import numpy as np

test_data = np.random.randn(40, 4) + 1
test_sample = Sampling(test_data)

test_sample.random()
