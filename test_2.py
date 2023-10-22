import numpy as np
import torch
from torch.nn import (
    Conv2d,
    ConvTranspose2d
)

import joatmon.nn.functional
from joatmon.nn.layer.conv import (
    Conv,
    ConvTranspose
)
from joatmon.nn.loss.mse import MSELoss

batch, channel, height, width = 1, 1, 5, 5

original_image = np.random.rand(batch, channel, height, width).astype('float32')
# original_image = np.expand_dims(original_image, axis=(0, 1))

in_channels = channel
out_channels = 3
kernel_height = 2
kernel_width = 2

stride = 1
padding = (0, 0)


def pretty(data):
    print('-' * 30)
    print(data)
    print('-' * 30)


def j(conv_weight, conv_bias, trans_weight, trans_bias):
    conv = Conv(in_channels, out_channels, (kernel_height, kernel_width), padding=(0, 0))
    conv.weight._data = conv_weight.detach().numpy()
    conv.bias._data = conv_bias.detach().numpy()

    trans = ConvTranspose(out_channels, in_channels, (kernel_height, kernel_width), padding=(0, 0))
    trans.weight._data = trans_weight.detach().numpy()
    trans.bias._data = trans_bias.detach().numpy()

    tensor = joatmon.nn.functional.from_array(original_image, requires_grad=True)
    pretty(tensor)

    downsampled = conv(tensor)
    pretty(downsampled)

    upsampled = trans(downsampled)
    pretty(upsampled)

    loss = MSELoss()
    mse = loss(upsampled, joatmon.nn.functional.zeros_like(upsampled))

    mse.backward()

    pretty(tensor.grad)
    pretty(conv.weight.grad)
    pretty(conv.bias.grad)
    pretty(trans.weight.grad)
    pretty(trans.bias.grad)


def t():
    conv = Conv2d(in_channels, out_channels, (kernel_height, kernel_width))
    trans = ConvTranspose2d(out_channels, in_channels, (kernel_height, kernel_width))

    tensor = torch.from_numpy(original_image)
    pretty(tensor)

    downsampled = conv(tensor)
    pretty(downsampled)

    upsampled = trans(downsampled)
    pretty(upsampled)

    loss = torch.nn.MSELoss()
    mse = loss(upsampled, torch.zeros_like(upsampled))

    mse.backward()

    pretty(tensor.grad)
    pretty(conv.weight.grad)
    pretty(conv.bias.grad)
    pretty(trans.weight.grad)
    pretty(trans.bias.grad)

    return conv.weight, conv.bias, trans.weight, trans.bias


params = t()
j(*params)
