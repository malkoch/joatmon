import json
import pickle

from joatmon.utility import JSONEncoder, to_case, to_enumerable


class Serializable(object):
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

    def __init__(self, **kwargs):
        for slot_name, slot_value in kwargs.items():
            if not isinstance(slot_name, str):
                raise ValueError(
                    f'{slot_name} is type {type(slot_name)} is not supported. only string type is supported for field names.'
                )
            setattr(self, slot_name, slot_value)

    def __str__(self):
        return self.pretty_json

    def __repr__(self):
        return str(self)

    def keys(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for key in self.__dict__.keys():
            yield key

    def values(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for value in self.__dict__.values():
            yield value

    def items(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for key, value in self.__dict__.items():
            yield key, value

    def __getitem__(self, key):
        return self.__dict__[key]

    @property
    def dict(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return to_enumerable(self)

    @classmethod
    def from_dict(cls, dictionary: dict):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return cls(**dictionary)

    @property
    def json(self) -> str:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return json.dumps(self.dict, cls=JSONEncoder)

    @property
    def pretty_json(self) -> str:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return json.dumps(self.dict, cls=JSONEncoder, indent=4)

    @classmethod
    def from_json(cls, string: str):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return cls.from_dict(json.loads(string))

    @property
    def bytes(self) -> bytes:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, value: bytes):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if value is None:
            return None

        obj = pickle.loads(value)
        return obj

    @property
    def snake(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return to_case('snake', self.__dict__)

    @property
    def pascal(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return to_case('pascal', self.__dict__)
