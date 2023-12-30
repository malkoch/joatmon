from joatmon.nn.core import *
from joatmon.nn.functional import *
from joatmon.nn.init import *
from joatmon.nn.layer import *
from joatmon.nn.loss import *
from joatmon.nn.optimizer import *
from joatmon.nn.scheduler import *

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
Tensor.log = log
Tensor.summation = summation
Tensor.mean = mean
Tensor.std = std
Tensor.var = var
Tensor.var = var
# Tensor.greater_or_equal = greater_or_equal
# Tensor.greater = greater
# Tensor.lesser_or_equal = lesser_or_equal
# Tensor.lesser = lesser
# Tensor.equal = equal
# Tensor.not_equal = not_equal
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
