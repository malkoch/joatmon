from __future__ import print_function

import webbrowser

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
            'name': 'browser',
            'description': 'a function for user to open a browser using a link',
            'parameters': {
                'type': 'object',
                'properties': {'url': {'type': 'string', 'description': 'url to open in the browser'}},
                'required': ['url'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        webbrowser.register(
            'firefox', None, webbrowser.BackgroundBrowser(r'C:\Program Files\Mozilla Firefox\firefox.exe')
        )
        webbrowser.get('firefox').open_new_tab(self.kwargs.get('url', None))

        if not self.stop_event.is_set():
            self.stop_event.set()
