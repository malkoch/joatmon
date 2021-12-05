from . import random
from .array import Array
from .functional import *

Array.astype = astype
Array.copy = copy
Array.repeat = repeat
Array.split = split
Array.tolist = tolist

Array.getitem = getitem
Array.take_along_axis = take_along_axis
Array.setitem = setitem
Array.put_along_axis = put_along_axis

Array.where = where
Array.indices = indices

Array.dim = dim
Array.size = size
Array.flatten = flatten
Array.reshape = reshape
Array.squeeze = squeeze
Array.expand_dims = expand_dims
Array.pad = pad
Array.transpose = transpose

Array.fill = fill

Array.abs = absolute
Array.negative = negative
Array.round = around
Array.floor = floor
Array.ceil = ceil
Array.sqrt = sqrt
Array.square = square
Array.clip = clip
Array.exp = exp
Array.tanh = tanh
Array.sum = sum
Array.mean = mean
Array.median = median
Array.var = var
Array.std = std
Array.prod = prod
Array.unique = unique
Array.argmax = argmax
Array.argmin = argmin
Array.amax = amax
Array.amin = amin

Array.add = add
Array.sub = sub
Array.truediv = truediv
Array.floordiv = floordiv
Array.mul = mul
Array.power = power
Array.lt = lt
Array.le = le
Array.gt = gt
Array.ge = ge
Array.eq = eq
Array.ne = ne
