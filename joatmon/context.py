__all__ = ['get_ctx', 'set_ctx', 'get_value', 'set_value']

context = None


def get_ctx():
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    global context
    return context


def set_ctx(ctx):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    global context
    context = ctx


def get_value(name):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    current = getattr(get_ctx(), 'current', {})
    return current.get(name, None)


def set_value(name, value):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    current = getattr(get_ctx(), 'current', {})
    if name not in current:
        current[name] = value
    setattr(get_ctx(), 'current', current)
