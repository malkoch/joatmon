import numpy as np
import torch

import joatmon.nn.functional
from joatmon.nn.loss.mse import MSELoss

batch, channel, height, width = 1, 1, 5, 5

original_image = np.random.rand(batch, channel, height, width).astype('float32')
# original_image = np.expand_dims(original_image, axis=(0, 1))

in_channels = channel
out_channels = 3
kernel_height = 2
kernel_width = 2

forward_weights = np.random.random((out_channels, in_channels, kernel_height, kernel_width)).astype('float32')
backward_weights = np.random.random((in_channels, out_channels, kernel_height, kernel_width)).astype('float32')
forward_bias = np.random.random((out_channels,)).astype('float32')
backward_bias = np.random.random((out_channels,)).astype('float32')
stride = 1
padding = (0, 0)

print('-' * 50)

my_tensor = joatmon.nn.functional.from_array(original_image, requires_grad=True)
my_forward_weights = joatmon.nn.functional.from_array(forward_weights, requires_grad=True)
my_backward_weights = joatmon.nn.functional.from_array(backward_weights, requires_grad=True)
my_forward_bias = joatmon.nn.functional.from_array(forward_bias, requires_grad=True)
my_backward_bias = joatmon.nn.functional.from_array(backward_bias, requires_grad=True)
print(my_tensor.shape)

tensor = torch.from_numpy(original_image)
tensor.requires_grad = True
forward_weights = torch.from_numpy(forward_weights)
forward_weights.requires_grad = True
backward_weights = torch.from_numpy(backward_weights)
backward_weights.requires_grad = True
forward_bias = torch.from_numpy(forward_bias)
forward_bias.requires_grad = True
backward_bias = torch.from_numpy(backward_bias)
backward_bias.requires_grad = True
print(tensor.shape)

print('-' * 50)

downsampled_image = joatmon.nn.functional.conv(my_tensor, weight=my_forward_weights, bias=my_forward_bias, stride=stride, padding=padding)
print(downsampled_image.shape)

downsampled = torch.nn.functional.conv2d(tensor, weight=forward_weights, bias=forward_bias, stride=stride, padding=padding)
print(downsampled.shape)

print('-' * 50)

upsampled_image = joatmon.nn.functional.conv_transpose(downsampled_image, weight=my_backward_weights, bias=my_backward_bias, stride=stride, padding=padding)
print(upsampled_image.shape)

upsampled = torch.nn.functional.conv_transpose2d(downsampled, weight=backward_weights, bias=backward_bias, stride=stride, padding=padding)
print(upsampled.shape)

print('-' * 50)

my_loss = MSELoss()
my_l = my_loss(upsampled_image, joatmon.nn.functional.zeros_like(upsampled_image))

my_l.backward()

loss = torch.nn.MSELoss()
mse = loss(upsampled, torch.zeros_like(upsampled))

mse.backward()

print(my_tensor.grad)
print(tensor.grad)
print('-' * 50)

print(my_forward_weights.grad)
print(forward_weights.grad)
print('-' * 50)

print(my_backward_weights.grad)
print(backward_weights.grad)
print('-' * 50)

print(my_forward_bias.grad)
print(forward_bias.grad)
print('-' * 50)

print(my_backward_bias.grad)
print(backward_bias.grad)
print('-' * 50)
