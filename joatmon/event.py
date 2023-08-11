import functools


class Event:
    def __init__(self):
        self._subscribers = {}

    def __iadd__(self, fn):
        func = fn
        runnable = fn

        if isinstance(fn, functools.partial):
            func = fn.func

        if func.__qualname__ not in self._subscribers:
            self._subscribers[func.__qualname__] = runnable
        else:
            # might want to rethink this
            raise ValueError(f'{func.__qualname__} already subscribed to this event')

        return self

    def __isub__(self, fn):
        if isinstance(fn, functools.partial):
            fn = fn.func

        if fn.__qualname__ in self._subscribers:
            del self._subscribers[fn.__qualname__]
        else:
            # might want to rethink this
            raise ValueError(f'{fn.__qualname__} already unsubscribed to this event')

        return self

    def fire(self, *args, **kwargs):
        for name, fn in self._subscribers.items():
            fn(*args, **kwargs)


class AsyncEvent:
    def __init__(self):
        self._subscribers = {}

    def __iadd__(self, fn):
        if fn.__qualname__ not in self._subscribers:
            self._subscribers[fn.__qualname__] = fn
        else:
            # might want to rethink this
            raise ValueError(f'{fn.__qualname__} already subscribed to this event')

        return self

    def __isub__(self, fn):
        if fn.__qualname__ in self._subscribers:
            del self._subscribers[fn.__qualname__]
        else:
            # might want to rethink this
            raise ValueError(f'{fn.__qualname__} already unsubscribed to this event')

        return self

    async def fire(self, *args, **kwargs):
        for name, fn in self._subscribers.items():
            await fn(*args, **kwargs)
