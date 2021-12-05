import math
from typing import (
    List,
    Optional,
    Tuple
)

import numpy as np

from .. import (
    functional as f,
    init
)
from ..core import (
    Module,
    Parameter,
    Tensor
)

__all__ = ['RNN', 'LSTM', 'GRU', 'RNNCell', 'LSTMCell', 'GRUCell']


# multilayer and multitime rnn grads does not work
class RNNBase(Module):
    __constants__ = ['mode', 'input_size', 'hidden_size', 'num_layers', 'bias', 'dropout']

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    dropout: float

    def __init__(self, mode: str, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True) -> None:
        super(RNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        if mode == 'LSTM':
            gate_size = 4 * hidden_size
        elif mode == 'GRU':
            gate_size = 3 * hidden_size
        elif mode == 'RNN_TANH':
            gate_size = hidden_size
        elif mode == 'RNN_RELU':
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._all_weights = []
        for layer in range(num_layers):
            w_ih = Parameter(Tensor.from_array(np.zeros((gate_size, input_size if layer == 0 else hidden_size))))
            w_hh = Parameter(Tensor.from_array(np.zeros((gate_size, hidden_size))))
            b_ih = Parameter(Tensor.from_array(np.zeros((gate_size,))))
            b_hh = Parameter(Tensor.from_array(np.zeros((gate_size,))))
            if bias:
                layer_params = (w_ih, w_hh, b_ih, b_hh)
            else:
                layer_params = (w_ih, w_hh)

            param_names = ['weight_ih_l{}', 'weight_hh_l{}']
            if bias:
                param_names += ['bias_ih_l{}', 'bias_hh_l{}']
            param_names = [x.format(layer) for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform(weight, -stdv, stdv)

    def check_input(self, inp: Tensor) -> None:
        expected_input_dim = 3
        if inp.dim() != expected_input_dim:
            raise RuntimeError('input must have {} dimensions, got {}'.format(expected_input_dim, inp.dim()))
        if self.input_size != inp.size(-1):
            raise RuntimeError('input.size(-1) must be equal to input_size. Expected {}, got {}'.format(self.input_size, inp.size(-1)))

    def get_expected_hidden_size(self, inp: Tensor) -> Tuple[int, int, int]:
        return self.num_layers, inp.size(0), self.hidden_size

    @staticmethod
    def check_hidden_size(hx: Tensor, expected_hidden_size: Tuple[int, int, int], msg: str = 'Expected hidden size {}, got {}') -> None:
        if hx.size() != expected_hidden_size:
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))

    def check_forward_args(self, inp: Tensor, hidden: Tensor):
        self.check_input(inp)
        expected_hidden_size = self.get_expected_hidden_size(inp)

        self.check_hidden_size(hidden, expected_hidden_size)

    def forward(self, inp: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        assert isinstance(inp, Tensor)
        max_batch_size = inp.size(0)

        if hx is None:
            hx = f.zeros((self.num_layers, max_batch_size, self.hidden_size), dtype=inp.dtype, device=inp.device)

        assert hx is not None
        self.check_forward_args(inp, hx)

        if self.nonlinearity == "tanh":
            ret = f.rnn_tanh(inp, hx, self.all_weights, self.bias, self.num_layers)
        elif self.nonlinearity == "relu":
            ret = f.rnn_relu(inp, hx, self.all_weights, self.bias, self.num_layers)
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))

        return ret

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if self.proj_size != 0:
            s += ', proj_size={proj_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(RNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']

        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(1):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}', 'weight_hr_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    if self.proj_size > 0:
                        self._all_weights += [weights]
                    else:
                        self._all_weights += [weights[:4]]
                else:
                    if self.proj_size > 0:
                        self._all_weights += [weights[:2]] + [weights[-1:]]
                    else:
                        self._all_weights += [weights[:2]]

    @property
    def all_weights(self) -> List[List[Parameter]]:
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]


class RNN(RNNBase):
    def __init__(self, *args, **kwargs):
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        if self.nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif self.nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(self.nonlinearity))
        super(RNN, self).__init__(mode, *args, **kwargs)


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__('LSTM', *args, **kwargs)

    def get_expected_cell_size(self, inp: Tensor) -> Tuple[int, int, int]:
        return self.num_layers, inp.size(0), self.hidden_size

    def check_forward_args(self, inp: Tensor, hidden: Tuple[Tensor, Tensor]):  # type: ignore
        self.check_input(inp)
        self.check_hidden_size(hidden[0], self.get_expected_hidden_size(inp), 'Expected hidden[0] size {}, got {}')
        self.check_hidden_size(hidden[1], self.get_expected_cell_size(inp), 'Expected hidden[1] size {}, got {}')

    def forward(self, inp, hx=None):
        if hx is None:
            h_zeros = f.zeros((self.num_layers, inp.size(0), self.hidden_size), dtype=inp.dtype, device=inp.device)
            c_zeros = f.zeros((self.num_layers, inp.size(0), self.hidden_size), dtype=inp.dtype, device=inp.device)
            hx = (h_zeros, c_zeros)

        self.check_forward_args(inp, hx)
        return f.lstm(inp, hx, self.all_weights, self.bias, self.num_layers)


class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        super(GRU, self).__init__('GRU', *args, **kwargs)

    def forward(self, inp, hx=None):
        max_batch_size = inp.size(0)

        if hx is None:
            hx = f.zeros((self.num_layers, max_batch_size, self.hidden_size), dtype=inp.dtype, device=inp.device)

        self.check_forward_args(inp, hx)
        return f.gru(inp, hx, self.all_weights, self.bias, self.num_layers)


class RNNCellBase(Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int) -> None:
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(Tensor(np.zeros((num_chunks * hidden_size, input_size))))
        self.weight_hh = Parameter(Tensor(np.zeros((num_chunks * hidden_size, hidden_size))))
        if bias:
            self.bias_ih = Parameter(Tensor(np.zeros((num_chunks * hidden_size))))
            self.bias_hh = Parameter(Tensor(np.zeros((num_chunks * hidden_size))))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, inp: Tensor) -> None:
        if inp.size(1) != self.input_size:
            raise RuntimeError("input has inconsistent input_size: got {}, expected {}".format(inp.size(1), self.input_size))

    def check_forward_hidden(self, inp: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if inp.size(0) != hx.size(0):
            raise RuntimeError("Input batch size {} doesn't match hidden{} batch size {}".format(inp.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError("hidden{} has inconsistent hidden_size: got {}, expected {}".format(hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform(weight, -stdv, stdv)


class RNNCell(RNNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'nonlinearity']
    nonlinearity: str

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh") -> None:
        super(RNNCell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def forward(self, inp: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(inp)
        if hx is None:
            hx = f.zeros((inp.size(0), self.hidden_size), dtype=inp.dtype, device=inp.device)
        self.check_forward_hidden(inp, hx, '')
        if self.nonlinearity == "tanh":
            ret = f.rnn_tanh_cell(inp, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        elif self.nonlinearity == "relu":
            ret = f.rnn_relu_cell(inp, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            raise RuntimeError("Unknown nonlinearity: {}".format(self.nonlinearity))
        return ret


class LSTMCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(LSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)

    def forward(self, inp: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        self.check_forward_input(inp)
        if hx is None:
            zeros = f.zeros((inp.size(0), self.hidden_size), dtype=inp.dtype, device=inp.device)
            hx = (zeros, zeros)
        self.check_forward_hidden(inp, hx[0], '[0]')
        self.check_forward_hidden(inp, hx[1], '[1]')
        return f.lstm_cell(inp, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)


class GRUCell(RNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(GRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)

    def forward(self, inp: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(inp)
        if hx is None:
            hx = f.zeros((inp.size(0), self.hidden_size), dtype=inp.dtype, device=inp.device)
        self.check_forward_hidden(inp, hx, '')
        return f.gru_cell(inp, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
