import pytest


def test__check_tensor_devices():
    assert True is True


def test__check_tensors():
    assert True is True


def test__create_tensor():
    assert True is True


def test__get_engine():
    assert True is True


def test__set_grad():
    assert True is True


def test_absolute():
    from joatmon.nn.functional import absolute
    from joatmon.nn.core import Tensor

    assert absolute(Tensor.from_array([0, -1])).data.tolist() == [0, 1]
    assert absolute(Tensor.from_array([0, 1])).data.tolist() == [0, 1]


def test_absolute_backward():
    from joatmon.nn.functional import absolute_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    absolute_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_adam():
    assert True is True


def test_add():
    from joatmon.nn.functional import add
    from joatmon.nn.core import Tensor

    assert add(Tensor.from_array([0, -1]), Tensor.from_array([0, 2])).data.tolist() == [0, 1]
    assert add(Tensor.from_array([0, -1]), 1).data.tolist() == [1, 0]


def test_add_backward():
    from joatmon.nn.functional import add_backward
    from joatmon.nn.core import Tensor

    inp1 = Tensor.from_array([1, 2])
    inp2 = Tensor.from_array([3, 4])
    add_backward(Tensor.from_array([1, 1]), inp1, inp2)

    assert inp1.grad.data.tolist() == [1, 1]
    assert inp2.grad.data.tolist() == [1, 1]


def test_arange():
    from joatmon.nn.functional import arange

    assert arange(1, 3, 1).data.tolist() == [1, 2]


def test_around():
    from joatmon.nn.functional import around
    from joatmon.nn.core import Tensor

    assert around(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_around_backward():
    from joatmon.nn.functional import around_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    around_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_avg_pool():
    assert True is True


def test_avg_pool_backward():
    assert True is True


def test_batch_norm():
    assert True is True


def test_batch_norm_backward():
    assert True is True


def test_ceil():
    from joatmon.nn.functional import ceil
    from joatmon.nn.core import Tensor

    assert ceil(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_ceil_backward():
    from joatmon.nn.functional import ceil_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    ceil_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_chunk():
    assert True is True


def test_chunk_backward():
    assert True is True


def test_clip():
    from joatmon.nn.functional import clip
    from joatmon.nn.core import Tensor

    assert clip(Tensor.from_array([0, -1]), -3, 3).data.tolist() == [0, -1]


def test_clip_backward():
    from joatmon.nn.functional import clip_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    clip_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_clone():
    from joatmon.nn.functional import clone
    from joatmon.nn.core import Tensor

    assert clone(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_clone_backward():
    from joatmon.nn.functional import clone_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    clone_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_concat():
    assert True is True


def test_concat_backward():
    assert True is True


def test_conv():
    assert True is True


def test_conv_backward():
    assert True is True


def test_cpu():
    from joatmon.nn.functional import cpu
    from joatmon.nn.core import Tensor

    assert cpu(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_dense():
    assert True is True


def test_dense_backward():
    assert True is True


def test_detach():
    from joatmon.nn.functional import detach
    from joatmon.nn.core import Tensor

    assert detach(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_div():
    from joatmon.nn.functional import div
    from joatmon.nn.core import Tensor

    assert div(Tensor.from_array([0, -2]), Tensor.from_array([1, 1])).data.tolist() == [0, -2]
    assert div(Tensor.from_array([0, -2]), 1).data.tolist() == [0, -2]


def test_div_backward():
    from joatmon.nn.functional import div_backward
    from joatmon.nn.core import Tensor

    inp1 = Tensor.from_array([3, 4])
    inp2 = Tensor.from_array([1, 2])
    div_backward(Tensor.from_array([1, 1]), inp1, inp2)

    assert inp1.grad.data.tolist() == [1, 0.5]
    assert inp2.grad.data.tolist() == [3, 4]


def test_double():
    from joatmon.nn.functional import double
    from joatmon.nn.core import Tensor

    assert double(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_dropout():
    assert True is True


def test_dropout_backward():
    assert True is True


def test_empty():
    from joatmon.nn.functional import empty

    assert empty(3).data.tolist() == [0, 0, 0]


def test_empty_like():
    from joatmon.nn.functional import empty_like
    from joatmon.nn.core import Tensor

    assert empty_like(Tensor.from_array([0, -1])).data.tolist() == [0, 0]


def test_expand_dim():
    assert True is True


def test_expand_dim_backward():
    assert True is True


def test_eye():
    from joatmon.nn.functional import eye

    assert eye(2, 2).data.tolist() == [[1, 0], [0, 1]]


def test_eye_like():
    from joatmon.nn.functional import eye_like
    from joatmon.nn.core import Tensor

    assert eye_like(Tensor.from_array([[1, 0], [0, 1]])).data.tolist() == [[1, 0], [0, 1]]


def test_fill():
    from joatmon.nn.functional import fill
    from joatmon.nn.core import Tensor

    assert fill(Tensor.from_array([0, -1]), 3).data.tolist() == [3, 3]


def test_floor():
    from joatmon.nn.functional import floor
    from joatmon.nn.core import Tensor

    assert floor(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_floor_backward():
    from joatmon.nn.functional import floor_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    floor_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_from_array():
    from joatmon.nn.functional import from_array

    assert from_array([0, -1]).data.tolist() == [0, -1]


def test_full():
    from joatmon.nn.functional import full

    assert full(3, 3).data.tolist() == [3, 3, 3]


def test_full_like():
    from joatmon.nn.functional import full_like
    from joatmon.nn.core import Tensor

    assert full_like(Tensor.from_array([0, -1]), 2).data.tolist() == [2, 2]


def test_gpu():
    from joatmon.nn.functional import gpu
    from joatmon.nn.core import Tensor

    assert gpu(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_half():
    from joatmon.nn.functional import half
    from joatmon.nn.core import Tensor

    assert half(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_index_select():
    assert True is True


def test_index_select_backward():
    assert True is True


def test_is_tensor():
    assert True is True


def test_linspace():
    from joatmon.nn.functional import linspace

    assert linspace(0, 3, 4).data.tolist() == [0, 1, 2, 3]


def test_lstm():
    assert True is True


def test_lstm_backward():
    assert True is True


def test_lstm_cell():
    assert True is True


def test_lstm_cell_backward():
    assert True is True


def test_max_pool():
    assert True is True


def test_max_pool_backward():
    assert True is True


def test_mean():
    from joatmon.nn.functional import mean
    from joatmon.nn.core import Tensor

    assert mean(Tensor.from_array([0, -2])).data.tolist() == -1


def test_mean_backward():
    from joatmon.nn.functional import mean_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    mean_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_mul():
    from joatmon.nn.functional import mul
    from joatmon.nn.core import Tensor

    assert mul(Tensor.from_array([0, -1]), Tensor.from_array([0, 2])).data.tolist() == [0, -2]
    assert mul(Tensor.from_array([0, -1]), 1).data.tolist() == [0, -1]


def test_mul_backward():
    from joatmon.nn.functional import mul_backward
    from joatmon.nn.core import Tensor

    inp1 = Tensor.from_array([1, 2])
    inp2 = Tensor.from_array([3, 4])
    mul_backward(Tensor.from_array([1, 1]), inp1, inp2)

    assert inp1.grad.data.tolist() == [3, 4]
    assert inp2.grad.data.tolist() == [1, 2]


def test_negative():
    from joatmon.nn.functional import negative
    from joatmon.nn.core import Tensor

    assert negative(Tensor.from_array([0, 1])).data.tolist() == [0, -1]


def test_negative_backward():
    from joatmon.nn.functional import negative_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    negative_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_normal():
    assert True is True


def test_normal_like():
    assert True is True


def test_one():
    from joatmon.nn.functional import one
    from joatmon.nn.core import Tensor

    assert one(Tensor.from_array([0, -1])).data.tolist() == [1, 1]


def test_ones():
    from joatmon.nn.functional import ones

    assert ones(2).data.tolist() == [1, 1]


def test_ones_like():
    from joatmon.nn.functional import ones_like
    from joatmon.nn.core import Tensor

    assert ones_like(Tensor.from_array([0, -1])).data.tolist() == [1, 1]


def test_power():
    from joatmon.nn.functional import power
    from joatmon.nn.core import Tensor

    assert power(Tensor.from_array([0, -1]), 1).data.tolist() == [0, -1]


def test_power_backward():
    from joatmon.nn.functional import power_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    power_backward(Tensor.from_array([1, 1]), inp, 1)

    assert inp.grad.data.tolist() == [1, 1]


def test_rand():
    assert True is True


def test_rand_like():
    assert True is True


def test_randint():
    assert True is True


def test_randint_like():
    assert True is True


def test_randn():
    assert True is True


def test_randn_like():
    assert True is True


def test_relu():
    from joatmon.nn.functional import relu
    from joatmon.nn.core import Tensor

    assert relu(Tensor.from_array([0, -1]), 0).data.tolist() == [0, 0]


def test_relu_backward():
    from joatmon.nn.functional import relu_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    relu_backward(Tensor.from_array([1, 1]), inp, 0)

    assert inp.grad.data.tolist() == [1, 1]


def test_rmsprop():
    assert True is True


def test_sigmoid():
    assert True is True


def test_sigmoid_backward():
    assert True is True


def test_single():
    from joatmon.nn.functional import single
    from joatmon.nn.core import Tensor

    assert single(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_softmax():
    assert True is True


def test_softmax_backward():
    assert True is True


def test_squeeze():
    assert True is True


def test_squeeze_backward():
    assert True is True


def test_stack():
    assert True is True


def test_stack_backward():
    assert True is True


def test_std():
    assert True is True


def test_std_backward():
    assert True is True


def test_sub():
    from joatmon.nn.functional import sub
    from joatmon.nn.core import Tensor

    assert sub(Tensor.from_array([0, -1]), Tensor.from_array([0, 2])).data.tolist() == [0, -3]
    assert sub(Tensor.from_array([0, -1]), 1).data.tolist() == [-1, -2]


def test_sub_backward():
    from joatmon.nn.functional import sub_backward
    from joatmon.nn.core import Tensor

    inp1 = Tensor.from_array([1, 2])
    inp2 = Tensor.from_array([3, 4])
    sub_backward(Tensor.from_array([1, 1]), inp1, inp2)

    assert inp1.grad.data.tolist() == [1, 1]
    assert inp2.grad.data.tolist() == [-1, -1]


def test_summation():
    from joatmon.nn.functional import summation
    from joatmon.nn.core import Tensor

    assert summation(Tensor.from_array([0, -1])).data.tolist() == -1


def test_summation_backward():
    from joatmon.nn.functional import summation_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    summation_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_tanh():
    assert True is True


def test_tanh_backward():
    assert True is True


def test_to_array():
    from joatmon.nn.functional import to_array
    from joatmon.nn.core import Tensor

    assert to_array(Tensor.from_array([0, -1])).tolist() == [0, -1]


def test_transpose():
    from joatmon.nn.functional import transpose
    from joatmon.nn.core import Tensor

    assert transpose(Tensor.from_array([0, -1])).data.tolist() == [0, -1]
    assert transpose(Tensor.from_array([[0, -1]])).data.tolist() == [[0], [-1]]


def test_transpose_backward():
    from joatmon.nn.functional import transpose_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    transpose_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_uniform():
    assert True is True


def test_uniform_like():
    assert True is True


def test_var():
    assert True is True


def test_var_backward():
    assert True is True


def test_view():
    from joatmon.nn.functional import view
    from joatmon.nn.core import Tensor

    assert view(Tensor.from_array([0, -1])).data.tolist() == [0, -1]


def test_view_backward():
    from joatmon.nn.functional import view_backward
    from joatmon.nn.core import Tensor

    inp = Tensor.from_array([1, 2])
    view_backward(Tensor.from_array([1, 1]), inp)

    assert inp.grad.data.tolist() == [1, 1]


def test_wrapped_partial():
    assert True is True


def test_zero():
    from joatmon.nn.functional import zero
    from joatmon.nn.core import Tensor

    assert zero(Tensor.from_array([0, -1])).data.tolist() == [0, 0]


def test_zeros():
    from joatmon.nn.functional import zeros

    assert zeros(2).data.tolist() == [0, 0]


def test_zeros_like():
    from joatmon.nn.functional import zeros_like
    from joatmon.nn.core import Tensor

    assert zeros_like(Tensor.from_array([0, -1])).data.tolist() == [0, 0]


if __name__ == '__main__':
    pytest.main([__file__])
