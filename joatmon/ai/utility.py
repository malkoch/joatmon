import os

import numpy as np
import torch


def load(network, path):
    if path is None or path == '':
        path = os.getcwd()
    if not os.path.exists(path):
        return

    network_path = os.path.join(path, f'network-weights.pth')

    if not os.path.exists(network_path):
        pass
    else:
        weights = torch.load(network_path)

        model_weights = network.state_dict()
        model_keys = list(model_weights.keys())

        for idx in range(len(model_keys)):
            try:
                model_key = model_keys[idx]
                key = model_key.replace('head.', '').replace('body.', '')

                model_weights[model_key] = weights[key]
            except Exception as ex:
                print(str(ex))

        network.load_state_dict(model_weights)


def save(network, path):
    if path is None or path == '':
        path = os.getcwd()
    if not os.path.exists(path):
        os.makedirs(path)

    network_path = os.path.join(path, f'network-weights.pth')

    weights = {}

    model_weights = network.state_dict()
    model_keys = list(model_weights.keys())

    for idx in range(len(model_keys)):
        try:
            model_key = model_keys[idx]
            key = model_key.replace('head.', '').replace('body.', '')

            weights[key] = model_weights[model_key]
        except Exception as ex:
            print(str(ex))

    torch.save(weights, network_path)


def display(values, positions):
    line = ''
    for i in range(len(values)):
        if i > 0:
            line = line[:-1] + ' '
        line += str(values[i])
        line = line[:positions[i]]
        line += ' ' * (positions[i] - len(line))
    print(line)


def easy_range(begin=0, end=None, increment=1):
    counter = begin
    while True:
        if end is not None:
            if counter > end:
                break
        yield counter
        counter += increment


def normalize(array, minimum=0.0, maximum=255.0, dtype='float32'):
    # nanmin and nanmax could be used
    array_minimum = float(np.amin(array))
    array_maximum = float(np.amax(array))
    # print(array_minimum, array_maximum, minimum, maximum)

    return np.asarray((array - array_minimum) * (maximum - minimum) / (array_maximum - array_minimum) + minimum, dtype=dtype)


def range_tensor(end):
    return torch.arange(end).long()


def to_numpy(t):
    return t.cpu().detach().numpy()


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.tensor(x, dtype=torch.float32)
    return x
