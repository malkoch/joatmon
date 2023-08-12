from __future__ import print_function

import os

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

    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return {
            'name': 'save',
            'description': 'a function for user to save a message to a file',
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string', 'description': 'content of the file'},
                    'path': {'type': 'string', 'description': 'path of the file'},
                },
                'required': ['message', 'path'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        path = self.kwargs.get('path', '') or self.api.input('what is the path')
        message = self.kwargs.get('message', '') or self.api.input('what is the message')

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        p = path if os.path.isabs(path) else os.path.join(parent_os_path, os_path[1:], path)

        with open(p, 'wb') as file:
            file.write(message.encode('utf-8', errors='ignore'))

        if not self.stop_event.is_set():
            self.stop_event.set()
