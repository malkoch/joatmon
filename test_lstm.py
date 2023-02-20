import numpy as np
import torch

from joatmon.ai.nn import functional as f
from joatmon.ai.nn.layer.lstm import LSTM


def test_1(input_size=1, hidden_size=2, sequence_length=4, layer_num=2, batch_size=12):
    inputs = np.random.randn(batch_size, sequence_length, input_size)
    output = np.random.randn(batch_size, sequence_length, hidden_size)
    lstm_size = (input_size, hidden_size, layer_num)

    t_lstm = torch.nn.LSTM(*lstm_size, batch_first=True)
    j_lstm = LSTM(*lstm_size)

    for t_layer, j_layer in zip(t_lstm._all_weights, j_lstm._all_weights):
        for t_weight, j_weight in zip(t_layer, j_layer):
            w = getattr(j_lstm, j_weight)
            setattr(w, '_data', getattr(t_lstm, t_weight).detach().numpy())

    t_inputs = torch.from_numpy(inputs.astype('float32'))
    j_inputs = f.from_array(inputs)

    t_out, t_hidden = t_lstm(t_inputs)
    j_out, j_hidden = j_lstm(j_inputs)

    print(t_out)
    print(j_out)
    print(np.allclose(t_out.detach().numpy(), j_out.data, rtol=1e-1, atol=1e-2))
    print()

    print(t_hidden[0])
    print(j_hidden[0])
    print(np.allclose(t_hidden[0].detach().numpy(), j_hidden[0].data, rtol=1e-1, atol=1e-2))
    print()

    print(t_hidden[1])
    print(j_hidden[1])
    print(np.allclose(t_hidden[1].detach().numpy(), j_hidden[1].data, rtol=1e-1, atol=1e-2))
    print()

    t_output = torch.from_numpy(output)
    j_output = f.from_array(output)

    t_loss = torch.sum((t_output - t_out) ** 2 / 2)
    j_loss = f.summation((j_output - j_out) ** 2 / 2)

    print(t_loss)
    print(j_loss)

    t_loss.backward()
    j_loss.backward()

    for t_layer, j_layer in zip(t_lstm._all_weights, j_lstm._all_weights):
        for t_weight, j_weight in zip(t_layer, j_layer):
            print('-' * 30)
            print(f't_weight {t_weight}: {getattr(t_lstm, t_weight).grad}')
            print(f'j_weight {j_weight}: {getattr(j_lstm, j_weight).grad}')
            print(np.allclose(getattr(t_lstm, t_weight).grad.detach().numpy(), getattr(j_lstm, j_weight).grad.data, rtol=1e-1, atol=1e-2))
            print('-' * 30)


test_1()
