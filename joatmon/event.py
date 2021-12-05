events = {}


def subscribe(name, callback):
    callbacks = events.get(name, [])
    callbacks.append(callback)
    events[name] = callbacks


def unsubscribe(name, callback):
    callbacks = events.get(name, [])
    while callback in callbacks:
        callbacks.remove(callback)
    events[name] = callbacks


def fire(name, data):
    for callback in events.get(name, []):
        callback(data)
