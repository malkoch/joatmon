import numpy as np
import torch

from joatmon.ai.nn import functional as f
from joatmon.ai.nn.layer.lstm import LSTM


def dense(inp, weight, bias):
    # return np.dot(inp, weight.T) + bias
    # print(inp.shape, weight.shape, bias.shape)
    return inp @ weight.T + bias


def sigmoid(inp):
    return 1 / (1 + np.exp(-inp))


def tanh(inp):
    return np.tanh(inp)


def lstm(inputs, hidden, weights, bias, num_layers):
    # inp.shape = b, t, f
    # hx.shape = n, b, h
    # cx.shape = n, b, h

    hx, cx = hidden
    out_tensor = np.zeros((inputs.shape[0], num_layers, inputs.shape[1], hx.shape[2]))
    hx_tensor = np.zeros((inputs.shape[0], num_layers, inputs.shape[1], hx.shape[2]))
    cx_tensor = np.zeros((inputs.shape[0], num_layers, inputs.shape[1], cx.shape[2]))
    for time in range(inputs.shape[1]):
        for layer in range(num_layers):
            if time == 0:
                if layer == 0:
                    cell_input = inputs[:, time, :]
                else:
                    cell_input = out_tensor[:, layer - 1, time, :]
            else:
                if layer == 0:
                    cell_input = inputs[:, time, :]
                else:
                    cell_input = out_tensor[:, layer - 1, time, :]

            if bias:
                w_ih, w_hh, b_ih, b_hh = weights[layer]
            else:
                w_ih, w_hh = weights[layer]
                b_ih, b_hh = None, None

            if time == 0:
                h = hx_tensor[:, layer, 0, :]
                c = cx_tensor[:, layer, 0, :]
            else:
                h = hx_tensor[:, layer, time - 1, :]
                c = cx_tensor[:, layer, time - 1, :]

            gates = cell_input @ w_ih.T + b_ih + h @ w_hh.T + b_hh

            ingate, forgetgate, cellgate, outgate = np.split(gates, 4, 1)

            ingate = sigmoid(ingate)
            forgetgate = sigmoid(forgetgate)
            cellgate = tanh(cellgate)
            outgate = sigmoid(outgate)

            c = (forgetgate * c) + (ingate * cellgate)
            h = outgate * tanh(c)

            hx_tensor[:, layer, time, :] = h
            cx_tensor[:, layer, time, :] = c

            out_tensor[:, layer, time, :] = h

    return out_tensor[:, num_layers - 1, :, :], (hx_tensor[:, :, inputs.shape[1] - 1, :], cx_tensor[:, :, inputs.shape[1] - 1, :])


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

    weights = []
    for t_layer in t_lstm._all_weights:
        layer_weights = []
        for t_weight in t_layer:
            layer_weights.append(getattr(t_lstm, t_weight).detach().numpy())
        weights.append(layer_weights)

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
    out, (h, c) = lstm(inputs, (np.zeros((layer_num, inputs.shape[0], hidden_size)), np.zeros((layer_num, inputs.shape[0], hidden_size))), weights, True, layer_num)

    print(t_out)
    print(j_out)
    # print(out)
    print(np.allclose(t_out.detach().numpy(), j_out.data, rtol=1e-1, atol=1e-2))
    # print(np.allclose(t_out.detach().numpy(), out, rtol=1e-1, atol=1e-2))
    print()

    print(t_hidden[0])
    print(j_hidden[0])
    # print(h)
    print(np.allclose(t_hidden[0].detach().numpy(), j_hidden[0].data, rtol=1e-1, atol=1e-2))
    # print(np.allclose(t_hidden[0].detach().numpy(), h, rtol=1e-1, atol=1e-2))
    print()

    print(t_hidden[1])
    print(j_hidden[1])
    # print(c)
    print(np.allclose(t_hidden[1].detach().numpy(), j_hidden[1].data, rtol=1e-1, atol=1e-2))
    # print(np.allclose(t_hidden[1].detach().numpy(), c, rtol=1e-1, atol=1e-2))
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
