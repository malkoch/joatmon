import functools


class Event:
    """
    Event class for managing event subscribers and firing events.

    This class provides a way to manage event subscribers and fire events.

    Attributes:
        _subscribers (dict): The subscribers of the event.
    """

    def __init__(self):
        self._subscribers = {}

    def __bool__(self):
        return bool(self._subscribers)

    def __iadd__(self, fn):
        """
        Add a function to the event subscribers.

        Args:
            fn (function): The function to add to the event subscribers.
        """
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
        """
        Remove a function from the event subscribers.

        Args:
            fn (function): The function to remove from the event subscribers.
        """
        if isinstance(fn, functools.partial):
            fn = fn.func

        if fn.__qualname__ in self._subscribers:
            del self._subscribers[fn.__qualname__]
        else:
            # might want to rethink this
            raise ValueError(f'{fn.__qualname__} already unsubscribed to this event')

        return self

    def fire(self, *args, **kwargs):
        """
        Fire the event to all subscribers.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ret = None
        for name, fn in self._subscribers.items():
            ret = fn(*args, **kwargs)
        return ret


class AsyncEvent:
    """
    AsyncEvent class for managing asynchronous event subscribers and firing events.

    This class provides a way to manage asynchronous event subscribers and fire events.

    Attributes:
        _subscribers (dict): The subscribers of the event.
    """

    def __init__(self):
        self._subscribers = {}

    def __bool__(self):
        return bool(self._subscribers)

    def __iadd__(self, fn):
        """
        Add a function to the event subscribers.

        Args:
            fn (function): The function to add to the event subscribers.
        """
        if fn.__qualname__ not in self._subscribers:
            self._subscribers[fn.__qualname__] = fn
        else:
            # might want to rethink this
            raise ValueError(f'{fn.__qualname__} already subscribed to this event')

        return self

    def __isub__(self, fn):
        """
        Remove a function from the event subscribers.

        Args:
            fn (function): The function to remove from the event subscribers.
        """
        if fn.__qualname__ in self._subscribers:
            del self._subscribers[fn.__qualname__]
        else:
            # might want to rethink this
            raise ValueError(f'{fn.__qualname__} already unsubscribed to this event')

        return self

    async def fire(self, *args, **kwargs):
        """
        Fire the event to all subscribers asynchronously.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        ret = None
        for name, fn in self._subscribers.items():
            ret = await fn(*args, **kwargs)
        return ret
