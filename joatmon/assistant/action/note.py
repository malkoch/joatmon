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
            'name': 'note',
            'description': 'a function for user to take note and list them afterwards',
            'parameters': {
                'type': 'object',
                'properties': {'mode': {'type': 'string', 'enum': ['create', 'list']}},
                'required': ['mode'],
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


class Service(BaseService):
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
