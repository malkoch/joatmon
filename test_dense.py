import numpy as np
import torch

from joatmon.ai.nn import functional as f
from joatmon.ai.nn.layer.linear import Linear


def test_1(input_size=2, output_size=4, batch_size=12):
    inputs = np.random.randn(batch_size, input_size)
    output = np.random.randn(batch_size, output_size)

    t_lstm = torch.nn.Linear(input_size, output_size)
    j_lstm = Linear(input_size, output_size)

    j_lstm.weight._data = t_lstm.weight.detach().numpy()
    j_lstm.bias._data = t_lstm.bias.detach().numpy()

    t_inputs = torch.from_numpy(inputs.astype('float32'))
    j_inputs = f.from_array(inputs)

    t_out = t_lstm(t_inputs)
    j_out = j_lstm(j_inputs)

    print(t_out)
    print(j_out)
    print(np.allclose(t_out.detach().numpy(), j_out.data, rtol=1e-1, atol=1e-2))
    print()

    t_output = torch.from_numpy(output)
    j_output = f.from_array(output)

    t_loss = torch.sum((t_output - t_out) ** 2 / 2)
    j_loss = f.summation((j_output - j_out) ** 2 / 2)

    print(t_loss)
    print(j_loss)

    t_loss.backward()
    j_loss.backward()

    print('-' * 30)
    print(f't_weight: {t_lstm.weight.grad}')
    print(f'j_weight: {t_lstm.weight.grad}')
    print('-' * 30)
    print('-' * 30)
    print(f't_weight: {t_lstm.bias.grad}')
    print(f'j_weight: {t_lstm.bias.grad}')
    print('-' * 30)


test_1()
