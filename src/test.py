from sampling import sampler as s
import numpy as np

test_data = np.random.randn(100, 5) + 1
test_sample = s.Sampler(test_data)

print(test_sample.systematic(sort='feature'))
