import numpy as np

# Create two arrays with shapes (2, 5, 1, 1) and (10, 5, 3, 4)
array1 = np.random.rand(2, 5, 1, 1)
array2 = np.random.rand(10, 5, 3, 4)

# Expand dimensions to match the desired output shape (2, 10, 3, 4)
array1 = array1[:, :, 0, 0]  # Shape becomes (2, 5)
array2 = array2[:, :, :, :]  # Shape becomes (10, 5, 3, 4)

# Multiply the arrays element-wise
result = array1[:, np.newaxis, np.newaxis, np.newaxis] * array2

# The resulting array has the shape (2, 10, 3, 4)
print(result.shape)
