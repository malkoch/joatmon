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
            'name': 'cd',
            'description': 'a function for user to change the current directory to given path',
            'parameters': {
                'type': 'object',
                'properties': {'path': {'type': 'string', 'description': 'the path that the user want to change to'}},
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
        path = self.kwargs.get('path', None) or self.api.input('what is path that you want to change')

        base = self.kwargs.get('base', '')
        cwd = self.kwargs.get('cwd', '')

        match path:
            case '.':
                ...
            case '..':
                if cwd != os.sep:
                    cwd = os.sep.join(cwd.split(os.sep)[:-1])
                    if cwd == '':
                        cwd = os.sep
            case _:
                cwd = os.path.join(cwd, path)

        self.api.cwd = cwd

        self.api.output(f'current path is: {cwd}')

        if not self.stop_event.is_set():
            self.stop_event.set()
