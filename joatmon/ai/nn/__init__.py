from joatmon.ai.nn.core import *
from joatmon.ai.nn.functional import *
from joatmon.ai.nn.init import *
from joatmon.ai.nn.layer import *
from joatmon.ai.nn.loss import *
from joatmon.ai.nn.optimizer import *
from joatmon.ai.nn.scheduler import *
# from joatmon.ai.nn.utility import *

Tensor.chunk = chunk
Tensor.view = view
Tensor.index_select = index_select

Tensor.zero = zero
Tensor.one = one
Tensor.fill = fill
Tensor.squeeze = squeeze
Tensor.expand_dim = expand_dim
Tensor.transpose = transpose
Tensor.absolute = absolute
Tensor.around = around
Tensor.floor = floor
Tensor.ceil = ceil
Tensor.clip = clip
Tensor.negative = negative
Tensor.summation = summation
Tensor.mean = mean
Tensor.std = std
Tensor.var = var
Tensor.var = var
Tensor.add = add
Tensor.sub = sub
Tensor.mul = mul
Tensor.div = div
Tensor.power = power
Tensor.clone = clone
Tensor.detach = detach

Tensor.from_array = from_array
Tensor.to_array = to_array

Tensor.half = half
Tensor.single = single
Tensor.double = double
Tensor.cpu = cpu
Tensor.gpu = gpu
