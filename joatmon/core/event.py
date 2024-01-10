import functools


class Event:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self):
        self._subscribers = {}

    def __bool__(self):
        return bool(self._subscribers)

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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ret = None
        for name, fn in self._subscribers.items():
            ret = fn(*args, **kwargs)
        return ret


class AsyncEvent:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self):
        self._subscribers = {}

    def __bool__(self):
        return bool(self._subscribers)

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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ret = None
        for name, fn in self._subscribers.items():
            ret = await fn(*args, **kwargs)
        return ret
