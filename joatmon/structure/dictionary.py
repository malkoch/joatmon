import json

from joatmon.core.utility import to_enumerable


def value_to_cls(value, cls):
    if isinstance(value, (list, tuple)):
        return [value_to_cls(v, cls) for v in value]
    elif isinstance(value, dict):
        return cls(value)
    else:
        return value


class CustomDictionary(dict):
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

    def __init__(self, data):  # ignore case parameter
        super(CustomDictionary, self).__init__()
        for k, v in data.items():
            self.__dict__[k] = value_to_cls(v, CustomDictionary)

    def __str__(self):
        return json.dumps(to_enumerable(self.__dict__))

    def __repr__(self):
        return str(self)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        for k, v in self.__dict__.items():
            yield k, v

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        item_parts = item.split('.')

        curr_item = None
        for idx, item_part in enumerate(item_parts):
            if idx == 0:
                curr_item = self.__dict__.get(item_part, None)
            else:
                if curr_item is None:
                    return curr_item
                else:
                    curr_item = curr_item[item_part]

        return curr_item

    def __setitem__(self, key, value):
        item_parts = key.split('.')

        if len(item_parts) == 1:
            self.__dict__[key] = value
            return

        curr_item = None
        for idx, item_part in enumerate(item_parts[:-1]):
            if idx == 0:
                curr_item = self.__dict__.get(item_part, None)
            else:
                if curr_item is None:
                    return
                else:
                    curr_item = curr_item[item_part]

        curr_item[item_parts[-1]] = value

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value
