__all__ = ['get_ctx', 'set_ctx', 'get_value', 'set_value']


class CTX:
    """
    CTX class for managing context.

    This class provides a way to manage context in a global scope.
    """


context = CTX()


def get_ctx():
    """
    Get the current context.

    This function returns the current context.

    Returns:
        CTX: The current context.
    """
    global context
    return context


def set_ctx(ctx):
    """
    Set the current context.

    This function sets the current context to the provided context.

    Args:
        ctx (CTX): The context to set.
    """
    global context
    context = ctx


def get_value(name):
    """
    Get a value from the current context.

    This function returns a value from the current context based on the provided name.

    Args:
        name (str): The name of the value to get.

    Returns:
        Any: The value from the current context, or None if the value does not exist.
    """
    current = getattr(get_ctx(), 'current', {})
    return current.get(name, None)


def set_value(name, value):
    """
    Set a value in the current context.

    This function sets a value in the current context based on the provided name and value.

    Args:
        name (str): The name of the value to set.
        value (Any): The value to set.
    """
    current = getattr(get_ctx(), 'current', {})
    if name not in current:
        current[name] = value
    setattr(get_ctx(), 'current', current)
