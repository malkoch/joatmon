from __future__ import print_function

import json
import os

import requests

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
            'name': 'exchange',
            'description': 'a function for user to learn exchange rates compared to TRY',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        key = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['exchange']['key']

        data = []

        resp = requests.get('https://api.twelvedata.com/exchange_rate', params={'apikey': key, 'symbol': 'USD/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        data.append(f'{response["symbol"]} : {response["rate"]}')

        resp = requests.get('https://api.twelvedata.com/exchange_rate', params={'apikey': key, 'symbol': 'EUR/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        data.append(f'{response["symbol"]} : {response["rate"]}')

        self.api.output('\n'.join(data))

        if not self.stop_event.is_set():
            self.stop_event.set()
