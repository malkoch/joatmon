__all__ = ['get_ctx', 'set_ctx', 'get_value', 'set_value']

context = None


def get_ctx():
    global context
    return context


def set_ctx(ctx):
    global context
    context = ctx


def get_value(name):
    current = getattr(get_ctx(), 'current', {})
    return current.get(name, None)


def set_value(name, value):
    current = getattr(get_ctx(), 'current', {})
    if name not in current:
        current[name] = value
    setattr(get_ctx(), 'current', current)
