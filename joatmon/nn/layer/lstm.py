import math
from typing import (
    List,
    Tuple
)

import numpy as np

from joatmon.nn import (
    functional as f,
    init
)
from joatmon.nn.core import (
    Module,
    Parameter,
    Tensor
)

__all__ = ['LSTM']


class LSTMCell(Module):
    """
    A Long Short Term Memory (LSTM) cell.

    # Arguments
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h

    # Attributes
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        weight_ih (Tensor): The learnable input-hidden weights, of shape (4*hidden_size x input_size)
        weight_hh (Tensor): The learnable hidden-hidden weights, of shape (4*hidden_size x hidden_size)
        bias_ih (Tensor): The learnable input-hidden bias, of shape (4*hidden_size)
        bias_hh (Tensor): The learnable hidden-hidden bias, of shape (4*hidden_size)
    """

    __constants__ = ['input_size', 'hidden_size']

    input_size: int
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        gate_size = 4 * hidden_size

        self._all_weights = ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']
        w_ih = Parameter(Tensor.from_array(np.zeros((gate_size, input_size))))
        w_hh = Parameter(Tensor.from_array(np.zeros((gate_size, hidden_size))))
        b_ih = Parameter(Tensor.from_array(np.zeros((gate_size,))))
        b_hh = Parameter(Tensor.from_array(np.zeros((gate_size,))))
        layer_params = (w_ih, w_hh, b_ih, b_hh)

        for name, param in zip(self._all_weights, layer_params):
            setattr(self, name, param)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the parameters (weight, bias) to their initial values.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform(weight, -stdv, stdv)

    def check_input(self, inp: Tensor) -> None:
        """
        Validates the input shape.

        # Arguments
            inp (Tensor): The input tensor.
        """
        expected_input_dim = 3
        if inp.dim() != expected_input_dim:
            raise RuntimeError('input must have {} dimensions, got {}'.format(expected_input_dim, inp.dim()))
        if self.input_size != inp.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(self.input_size, inp.size(-1))
            )

    def get_expected_hidden_size(self, inp: Tensor) -> Tuple[int, int]:
        """
        Returns the expected hidden state size.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tuple[int, int]: The expected hidden state size.
        """
        return inp.size(0), self.hidden_size

    def extra_repr(self) -> str:
        """
        Returns a string containing a brief description of the module.

        # Returns
            str: A string containing a brief description of the module.
        """
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(LSTMCell, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']

        if isinstance(self._all_weights[0], str):
            return

        self._all_weights = ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']

    @property
    def all_weights(self) -> List[List[Parameter]]:
        """
        Returns a list of all weights for the LSTM cell.

        # Returns
            List[List[Parameter]]: A list of all weights for the LSTM cell.
        """
        return [getattr(self, weights) for weights in self._all_weights]

    def get_expected_cell_size(self, inp: Tensor) -> Tuple[int, int]:
        """
        Returns the expected cell state size.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tuple[int, int]: The expected cell state size.
        """
        return inp.size(0), self.hidden_size

    def check_forward_args(self, inp: Tensor):  # type: ignore
        """
        Validates the input shape for the forward pass.

        # Arguments
            inp (Tensor): The input tensor.
        """
        self.check_input(inp)

    def forward(self, inp):
        """
        Defines the computation performed at every call.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying LSTM cell.
        """
        self.check_forward_args(inp)
        return f.lstm(inp, self.all_weights)


class LSTM(Module):
    """
    A Long Short Term Memory (LSTM) module.

    # Arguments
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        num_layers (int, optional): Number of recurrent layers. Default: 1

    # Attributes
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
        num_layers (int): Number of recurrent layers.
    """

    __constants__ = ['input_size', 'hidden_size', 'num_layers']

    input_size: int
    hidden_size: int
    num_layers: int

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        gate_size = 4 * hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            w_ih = Parameter(Tensor.from_array(np.zeros((gate_size, input_size if layer == 0 else hidden_size))))
            w_hh = Parameter(Tensor.from_array(np.zeros((gate_size, hidden_size))))
            b_ih = Parameter(Tensor.from_array(np.zeros((gate_size,))))
            b_hh = Parameter(Tensor.from_array(np.zeros((gate_size,))))
            layer_params = (w_ih, w_hh, b_ih, b_hh)

            param_names = ['weight_ih_l{}', 'weight_hh_l{}']
            param_names += ['bias_ih_l{}', 'bias_hh_l{}']
            param_names = [x.format(layer) for x in param_names]

            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets the parameters (weight, bias) to their initial values.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform(weight, -stdv, stdv)

    def check_input(self, inp: Tensor) -> None:
        """
        Validates the input shape.

        # Arguments
            inp (Tensor): The input tensor.
        """
        expected_input_dim = 3
        if inp.dim() != expected_input_dim:
            raise RuntimeError('input must have {} dimensions, got {}'.format(expected_input_dim, inp.dim()))
        if self.input_size != inp.size(-1):
            raise RuntimeError(
                'input.size(-1) must be equal to input_size. Expected {}, got {}'.format(self.input_size, inp.size(-1))
            )

    def get_expected_hidden_size(self, inp: Tensor) -> Tuple[int, int, int]:
        """
        Returns the expected hidden state size.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tuple[int, int, int]: The expected hidden state size.
        """
        return self.num_layers, inp.size(0), self.hidden_size

    def extra_repr(self) -> str:
        """
        Returns a string containing a brief description of the module.

        # Returns
            str: A string containing a brief description of the module.
        """
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(LSTM, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']

        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        self._all_weights = []
        for layer in range(num_layers):
            weights = ['weight_ih_l{}', 'weight_hh_l{}', 'bias_ih_l{}', 'bias_hh_l{}']
            weights = [x.format(layer) for x in weights]
            self._all_weights += [weights[:4]]

    @property
    def all_weights(self) -> List[List[Parameter]]:
        """
        Returns a list of all weights for the LSTM module.

        # Returns
            List[List[Parameter]]: A list of all weights for the LSTM module.
        """
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    def get_expected_cell_size(self, inp: Tensor) -> Tuple[int, int, int]:
        """
        Returns the expected cell state size.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tuple[int, int, int]: The expected cell state size.
        """
        return self.num_layers, inp.size(0), self.hidden_size

    def check_forward_args(self, inp: Tensor):  # type: ignore
        """
        Validates the input shape for the forward pass.

        # Arguments
            inp (Tensor): The input tensor.
        """
        self.check_input(inp)

    def forward(self, inp):
        """
        Defines the computation performed at every call.

        # Arguments
            inp (Tensor): The input tensor.

        # Returns
            Tensor: The output tensor after applying LSTM.
        """
        self.check_forward_args(inp)
        return f.lstm(inp, self.all_weights)
