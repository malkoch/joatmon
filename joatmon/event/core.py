class Event:
    def __init__(self):
        self._subscribers = {}

    def __iadd__(self, fn):
        if fn.__qualname__ not in self._subscribers:
            self._subscribers[fn.__qualname__] = fn
        else:
            raise ValueError(f'{fn.__qualname__} already subscribed to this event')

        return self

    def __isub__(self, fn):
        if fn.__qualname__ in self._subscribers:
            del self._subscribers[fn.__qualname__]
        else:
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
            raise ValueError(f'{fn.__qualname__} already subscribed to this event')

        return self

    def __isub__(self, fn):
        if fn.__qualname__ in self._subscribers:
            del self._subscribers[fn.__qualname__]
        else:
            raise ValueError(f'{fn.__qualname__} already unsubscribed to this event')

        return self

    async def fire(self, *args, **kwargs):
        for name, fn in self._subscribers.items():
            await fn(*args, **kwargs)
