__all__ = []


def prod(inp, axis=None) -> int:
    """
    Calculate the product of all elements in the input list.

    Args:
        inp (list): The input list.
        axis (None): This argument is not used.

    Returns:
        int: The product of all elements in the input list.
    """
    p = 1
    for data in inp:
        p *= data
    return p


def unravel_index(index: int, shape: list) -> list:
    """
    Convert a flat index into a multi-dimensional index.

    Args:
        index (int): The flat index.
        shape (list): The shape of the multi-dimensional array.

    Returns:
        list: The multi-dimensional index.
    """
    r = []
    for i in range(len(shape)):
        div, mod = divmod(index, prod(shape[i + 1:]))
        r.append(div)
        index = mod
    return r


def ravel_index(index: list, shape: list) -> int:
    """
    Convert a multi-dimensional index into a flat index.

    Args:
        index (list): The multi-dimensional index.
        shape (list): The shape of the multi-dimensional array.

    Returns:
        int: The flat index.
    """
    r = 0
    for i, _idx in enumerate(index):
        r += prod(shape[i + 1:]) * _idx
    return r


def dim(inp: list) -> int:
    """
    Calculate the dimensionality of the input list.

    Args:
        inp (list): The input list.

    Returns:
        int: The dimensionality of the input list.
    """
    if isinstance(inp, (bool, int, float)):
        return 0
    return dim(inp[0]) + 1


def flatten(inp: list) -> list:
    """
    Flatten a multi-dimensional list into a one-dimensional list.

    Args:
        inp (list): The multi-dimensional list.

    Returns:
        list: The flattened list.
    """
    data_type = type(inp)

    ret = []
    for data in inp:
        if dim(data) == 0:
            ret.append(data)
        elif dim(data) == 1:
            ret += data
        else:
            ret += flatten(data)
    return data_type(ret)


def size(inp: list, axis: int = None):
    """
    Calculate the size of each dimension of the input list.

    Args:
        inp (list): The input list.
        axis (int, optional): If provided, calculate the size only along this dimension.

    Returns:
        tuple: The size of each dimension of the input list.
    """
    if isinstance(inp, (bool, int, float)):
        return ()

    if axis is None or axis < 0:
        return tuple([len(inp)] + list(size(inp[0])))
    return tuple((size(inp[0], axis=axis - 1)))


def reshape(inp: list, shape: list) -> list:
    """
    Reshape a flat list into a multi-dimensional list.

    Args:
        inp (list): The flat list.
        shape (list): The shape of the multi-dimensional list.

    Returns:
        list: The reshaped list.
    """
    flat = flatten(inp)

    subdims = shape[1:]
    subsize = prod(subdims)
    if shape[0] * subsize != len(flat):
        raise ValueError('size does not match or invalid')
    if not subdims:
        return flat
    return [reshape(flat[i: i + subsize], subdims) for i in range(0, len(flat), subsize)]
