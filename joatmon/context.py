import asyncio
from functools import partial
from threading import get_ident


class LocalProxy(object):
    def __init__(self, func):
        self.func = func

    def __setitem__(self, key, value):
        self.func()[key] = value

    def __getitem__(self, key):
        return self.func().get(key, None)

    def __delitem__(self, key):
        ...

    def get(self, key, default):
        return self.func().get(key, default)


def lookup_object(name):
    try:
        task = asyncio.current_task(asyncio.get_running_loop())
        context = task.context
    except RuntimeError:
        task = get_ident()
        context = _stack.get(task, None)
    return context[name]


def initialize_context():
    try:
        task = asyncio.current_task(asyncio.get_running_loop())
        context = {'current': {}}
        task.context = context
    except RuntimeError:
        task = get_ident()
        context = {'current': {}}
        _stack[task] = context


def teardown_context():
    try:
        task = asyncio.current_task(asyncio.get_running_loop())
        del task.context
    except RuntimeError:
        task = get_ident()
        del _stack[task]


_stack = {}
current = LocalProxy(partial(lookup_object, name='current'))
