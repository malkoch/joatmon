import json
import os

import numpy as np

from joatmon.nn import functional


def load(network, path):
    """
    Load the weights of a network from a file.

    Args:
        network (nn.Module): The PyTorch network for which the weights are loaded.
        path (str): The path to the directory containing the weights file.
    """
    if path is None or path == '':
        path = os.getcwd()
    if not os.path.exists(path):
        return

    network_path = os.path.join(path, f'network-weights.pth')

    if not os.path.exists(network_path):
        pass
    else:
        weights = json.load(open(network_path, 'r'))

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
    """
    Save the weights of a network to a file.

    Args:
        network (nn.Module): The PyTorch network for which the weights are saved.
        path (str): The path to the directory where the weights file will be saved.
    """
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

    json.dump(weights, open(network_path, 'w'))


def display(values, positions):
    """
    Display a list of values in a formatted string.

    Args:
        values (list): The list of values to be displayed.
        positions (list): The list of positions where each value should be displayed in the string.
    """
    line = ''
    for i in range(len(values)):
        if i > 0:
            line = line[:-1] + ' '
        line += str(values[i])
        line = line[: positions[i]]
        line += ' ' * (positions[i] - len(line))
    print(line)


def easy_range(begin=0, end=None, increment=1):
    """
    Generate a range of numbers.

    Args:
        begin (int): The number at which the range begins.
        end (int): The number at which the range ends.
        increment (int): The increment between each number in the range.

    Returns:
        generator: A generator that yields the numbers in the range.
    """
    counter = begin
    while True:
        if end is not None:
            if counter > end:
                break
        yield counter
        counter += increment


def normalize(array, minimum=0.0, maximum=255.0, dtype='float32'):
    """
    Normalize an array to a specified range.

    Args:
        array (numpy array): The array to be normalized.
        minimum (float): The minimum value of the range.
        maximum (float): The maximum value of the range.
        dtype (str): The data type of the normalized array.

    Returns:
        numpy array: The normalized array.
    """
    array_minimum = float(np.amin(array))
    array_maximum = float(np.amax(array))

    return np.asarray(
        (array - array_minimum) * (maximum - minimum) / (array_maximum - array_minimum) + minimum, dtype=dtype
    )


def range_tensor(end):
    """
    Create a tensor with a range of numbers.

    Args:
        end (int): The number at which the range ends.

    Returns:
        Tensor: A tensor containing the numbers in the range.
    """
    return functional.arange(end).long()


def to_numpy(t):
    """
    Convert a tensor to a numpy array.

    Args:
        t (Tensor): The tensor to be converted.

    Returns:
        numpy array: The converted numpy array.
    """
    return t.cpu().detach().numpy()


def to_tensor(x):
    """
    Convert a value to a tensor.

    Args:
        x (various types): The value to be converted.

    Returns:
        Tensor: The converted tensor.
    """
    if isinstance(x, functional.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = functional.tensor(x, dtype=functional.float32)
    return x
