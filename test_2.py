import numpy as np
import torch

import joatmon.nn.functional

original_image = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    dtype='float32'
)
original_image = np.expand_dims(original_image, axis=(0, 1))

weights = np.random.random((5, 1, 2, 2)).astype('float32')
bias = np.random.random((5,)).astype('float32')
stride = 1
padding = (0, 0)

my_tensor = joatmon.nn.functional.from_array(original_image)
my_weights = joatmon.nn.functional.from_array(weights)
my_bias = joatmon.nn.functional.from_array(bias)
print(my_tensor)

tensor = torch.from_numpy(original_image)
weights = torch.from_numpy(weights).transpose(0, 1)
bias = torch.from_numpy(bias)
print(tensor)

print('-' * 50)

upsampled_image = joatmon.nn.functional.conv_transpose(my_tensor, weight=my_weights, bias=my_bias, stride=stride, padding=padding)
print(upsampled_image)

upsampled = torch.nn.functional.conv_transpose2d(tensor, weight=weights, bias=bias, stride=stride, padding=padding)
print(upsampled)

print('-' * 50)
