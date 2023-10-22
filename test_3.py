import numpy as np
import torch

import joatmon.nn.functional

scale_factor = 2
original_image = np.array(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    dtype='float32'
)
original_image = np.expand_dims(original_image, axis=(0, 1))

my_tensor = joatmon.nn.functional.from_array(original_image, requires_grad=True)
print(my_tensor)

tensor = torch.from_numpy(original_image)
tensor.requires_grad = True
print(tensor)

print('-' * 50)

upsampled_image = joatmon.nn.functional.bilinear_interpolation(my_tensor, scale_factor)
print(upsampled_image)

upsampled = torch.nn.functional.interpolate(tensor, scale_factor=scale_factor, mode='bilinear')
print(upsampled)

print('-' * 50)

joatmon.nn.functional.summation(upsampled_image).backward()
print(my_tensor.grad)

upsampled.sum().backward()
print(tensor.grad)
