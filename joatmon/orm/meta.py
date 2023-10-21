import inspect
import typing

from joatmon.orm.constraint import Constraint
from joatmon.orm.field import Field
from joatmon.orm.index import Index
from joatmon.core.utility import get_converter


class Meta(type):
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

    __collection__ = 'meta'

    structured = True
    force = True

    qb = None

    def __new__(mcs, name, bases, dct):
        return super().__new__(mcs, name, bases, dct)

    def fields(cls, predicate=lambda x: True) -> typing.Dict[str, Field]:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Field)) if predicate(v)}

    def constraints(cls, predicate=lambda x: True) -> typing.Dict[str, Constraint]:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Constraint)) if predicate(v)}

    def indexes(cls, predicate=lambda x: True) -> typing.Dict[str, Index]:
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {k: v for k, v in inspect.getmembers(cls, lambda x: isinstance(x, Index)) if predicate(v)}

    def query(cls):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return cls.qb


def normalize_kwargs(meta, **kwargs):
    """
    Remember the transaction.

    Accepts a state, action, reward, next_state, terminal transaction.

    # Arguments
        transaction (abstract): state, action, reward, next_state, terminal transaction.
    """
    ret = {}

    fields = meta.fields(meta)
    for key in kwargs.keys():
        field = list(filter(lambda x: x[0] == key, fields.items()))
        if len(field) != 1:
            raise ValueError(f'field {key} has to be only one on the document')
        field = field[0][1]

        ret[key] = get_converter(field.dtype)(kwargs[key])
    return ret
