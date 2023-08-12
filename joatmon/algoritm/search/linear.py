def linear_search(iterable, value, item_getter):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    for item in iterable:
        if item_getter(item) == value:
            return item
