from __future__ import print_function

import os
import shutil

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
            'name': 'move',
            'description': 'a function for user to move a file from one directory to another',
            'parameters': {
                'type': 'object',
                'properties': {
                    'old': {'type': 'string', 'description': 'path of the file that want to be moved'},
                    'new': {'type': 'string', 'description': 'new path of the file'},
                },
                'required': ['old', 'new'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        old_file = self.kwargs.get('old', '') or self.api.input('what is the old file')
        new_file = self.kwargs.get('new', '') or self.api.input('what is the new file')

        parent_os_path = self.kwargs.get('parent_os_path', '')
        os_path = self.kwargs.get('os_path', '')

        shutil.move(
            os.path.join(parent_os_path, os_path[1:], old_file), os.path.join(parent_os_path, os_path[1:], new_file)
        )

        if not self.stop_event.is_set():
            self.stop_event.set()
