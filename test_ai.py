import torch
import numpy as np
from joatmon.ai.nn import functional as f
from joatmon.ai.nn.layer.lstm import LSTM


def test_1(input_size=1, hidden_size=2, sequence_length=3, layer_num=4, batch_size=5):
    inputs = np.random.randn(batch_size, sequence_length, input_size)
    lstm_size = (input_size, hidden_size, layer_num)

    t_lstm = torch.nn.LSTM(*lstm_size)
    j_lstm = LSTM(*lstm_size)

    for t_layer, j_layer in zip(t_lstm._all_weights, j_lstm._all_weights):
        for t_weight, j_weight in zip(t_layer, j_layer):
            w = getattr(j_lstm, j_weight)
            setattr(w, '_data', getattr(t_lstm, t_weight).detach().numpy())

    # for t_layer, j_layer in zip(t_lstm._all_weights, j_lstm._all_weights):
    #     for t_weight, j_weight in zip(t_layer, j_layer):
    #         t_w = getattr(t_lstm, t_weight)
    #         j_w = getattr(j_lstm, j_weight)
    #         print(t_w)
    #         print(j_w)
    #         print()
    #
    # print(t_lstm._all_weights)
    # print(j_lstm._all_weights)

    t_inputs = torch.from_numpy(inputs.astype('float32'))
    j_inputs = f.from_array(inputs)

    # print(t_inputs)
    # print(j_inputs)
    # print()

    t_out, t_hidden = t_lstm(t_inputs)
    j_out, j_hidden = j_lstm(j_inputs)

    print(t_out)
    print(j_out)
    print()

    print(t_hidden[0])
    print(j_hidden[0])
    print()

    print(t_hidden[1])
    print(j_hidden[1])
    print()


test_1()
