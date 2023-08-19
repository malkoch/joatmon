from __future__ import print_function

import json
import os

import openai

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
            'name': 'translate',
            'description': 'a function for user to translate a message',
            'parameters': {
                'type': 'object',
                'properties': {'message': {'type': 'string', 'description': 'message to be translated'}},
                'required': ['message'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        config = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['openai']
        openai.api_key = config['key']

        message = self.kwargs.get('message', None) or self.api.input('what do you want the message to be')

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {
                    'role': 'user',
                    'content': 'from now on you are going to translate my input and generate only the translated message',
                },
                {
                    'role': 'agent',
                    'content': 'Understood! From now on, I will translate your input and provide the translated message. '
                               'Please go ahead and let me know what you '
                               'would like to be translated.',
                },
                {'role': 'user', 'content': message},
            ],
        )
        self.api.output(response['choices'][0]['message']['content'])

        if not self.stop_event.is_set():
            self.stop_event.set()
