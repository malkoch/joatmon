import numpy as np

arr1 = np.random.randn(5, 2)
arr2 = np.random.randn(5, 2)

print(np.stack([arr1, arr2], axis=0).shape)
