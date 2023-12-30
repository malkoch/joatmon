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
            'name': 'rm',
            'description': 'a function for user to remove a file or a folder with its content',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string', 'description': 'path to remove'}},
                'required': ['path'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        path = self.kwargs.get('path', '') or self.api.input('which folder/file do you want to remove')

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        if os.path.isdir(os.path.join(parent_os_path, os_path[1:], path)):
            os.removedirs(os.path.join(parent_os_path, os_path[1:], path))
        else:
            os.remove(os.path.join(parent_os_path, os_path[1:], path))

        if not self.stop_event.is_set():
            self.stop_event.set()
