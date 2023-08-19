from __future__ import print_function

import os.path
import subprocess

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
            'name': 'run',
            'description': 'a function for user to run an executable',
            'parameters': {
                'type': 'object',
                'properties': {'executable': {'type': 'string', 'description': 'executable to run'}},
                'required': ['executable'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        executable = self.kwargs.get('executable', '') or self.api.input('what do you want to run')
        args = self.kwargs.get('args', '')

        # send the os path to all executables
        # send the parent os path to all executables

        executable_path = os.path.join(self.kwargs.get('base', '/'), 'executables', executable)

        subprocess.run(['python.exe', executable_path] + args.split(' ') + ['--task', self.name])

        if not self.stop_event.is_set():
            self.stop_event.set()
