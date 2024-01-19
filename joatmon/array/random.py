def rand(shape):
    """
    Generate a random array of a given shape.

    This function generates an array of the given shape and populates it with random samples from a uniform distribution over [0, 1).

    Args:
        shape (tuple): The shape of the output array.

    Returns:
        array: An array of the given shape filled with random samples.
    """


def randint(low, high, shape):
    """
    Generate a random integer array of a given shape.

    This function generates an array of the given shape and populates it with random integers from `low` (inclusive) to `high` (exclusive).

    Args:
        low (int): The lowest (signed) integer to be drawn from the distribution.
        high (int): One above the largest (signed) integer to be drawn from the distribution.
        shape (tuple): The shape of the output array.

    Returns:
        array: An array of the given shape filled with random integers.
    """


def randn(shape):
    """
    Generate a random array of a given shape.

    This function generates an array of the given shape and populates it with random samples from a standard Normal distribution (mean=0, stdev=1).

    Args:
        shape (tuple): The shape of the output array.

    Returns:
        array: An array of the given shape filled with random samples.
    """


def normal(loc, scale, shape):
    """
    Generate a random array of a given shape.

    This function generates an array of the given shape and populates it with random samples from a Normal distribution with mean `loc` and standard deviation `scale`.

    Args:
        loc (float): The mean (“centre”) of the distribution.
        scale (float): The standard deviation (spread or “width”) of the distribution.
        shape (tuple): The shape of the output array.

    Returns:
        array: An array of the given shape filled with random samples.
    """


def uniform(low, high, shape):
    """
    Generate a random array of a given shape.

    This function generates an array of the given shape and populates it with random samples from a uniform distribution over [low, high).

    Args:
        low (float): Lower boundary of the output interval. All values generated will be greater than or equal to low.
        high (float): Upper boundary of the output interval. All values generated will be less than high.
        shape (tuple): The shape of the output array.

    Returns:
        array: An array of the given shape filled with random samples.
    """
