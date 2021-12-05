from joatmon.context import current

plugins = {}


# if creator is string import it, if any of args or kwargs are callable call them
def register(name, creator, *args, **kwargs):
    plugins[name] = (creator, args, kwargs)


def unregister(name):
    plugins.pop(name, None)


def create(name):  # let user pass extra parameters
    creator, args, kwargs = plugins[name]
    obj = creator(name, *args, **kwargs)
    return obj


class Plugin:
    def __init__(self, alias):
        self.alias = alias

    def __enter__(self):
        current[self.alias] = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del current[self.alias]
