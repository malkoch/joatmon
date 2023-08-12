def normal(param, loc=0.0, scale=1.0):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    import numpy as np

    param._data = np.random.normal(loc, scale, size=param.shape)


def uniform(param, low=-1.0, high=1.0):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    import numpy as np

    param._data = np.random.uniform(low, high, size=param.shape)
