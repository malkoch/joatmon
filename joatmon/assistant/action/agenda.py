from __future__ import print_function

from joatmon.assistant.service import BaseService
from joatmon.assistant.task import BaseTask


class Task(BaseTask):
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

    def __init__(self, name, api, **kwargs):
        super(Task, self).__init__(name, api, **kwargs)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {
            'name': 'agenda',
            'description': 'a function for user to create, list, delete, update, search an event in their agenda',
            'parameters': {
                'type': 'object',
                'properties': {
                    'mode': {'type': 'string', 'enum': ['create', 'list', 'delete', 'update', 'search']},
                    'event': {'type': 'string', 'description': 'the event name'},
                    'date': {'type': 'string', 'description': 'datetime of the event'},
                },
                'required': ['mode', 'event'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        mode = self.kwargs.get('mode', None)

        if mode == 'list':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'create':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'delete':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'update':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))
        if mode == 'search':
            print(self.kwargs.get('mode', None), self.kwargs.get('event', None), self.kwargs.get('date', None))


class Service(BaseService):
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

    def __init__(self, name, api, **kwargs):
        super(Service, self).__init__(name, api, **kwargs)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {}

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...
