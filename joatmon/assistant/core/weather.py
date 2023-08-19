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
            'name': 'weather',
            'description': 'a function for user to learn current, historical, forecast weather data for given location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'mode': {'type': 'string', 'enum': ['current', 'history', 'forecast']},
                    'location': {'type': 'string', 'description': 'location of the weather'},
                },
                'required': ['mode', 'location'],
            },
        }

    def run(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        key = json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['weather']['key']

        mode = self.kwargs.get('mode', None)
        location = self.kwargs.get('location', None) or self.api.input('what is the location')

        if mode == 'current':
            resp = requests.get('https://api.weatherapi.com/v1/current.json', params={'key': key, 'q': location})

            response = json.loads(resp.content.decode('utf-8'))
            current = response.get('current', {})
            temp_c = current.get('temp_c', None)
            current.get('condition', {}).get('text', None)
            current.get('wind_kph', None)
            current.get('wind_degree', None)
            current.get('wind_dir', None)
            current.get('pressure_mb', None)
            humidity = current.get('humidity', None)
            current.get('cloud', None)
            feelslike_c = current.get('feelslike_c', None)
            current.get('vis_km', None)

            self.api.output(
                f'Current Temperature is: {temp_c} and it feels like {feelslike_c}, Current Humidity is: {humidity}'
            )

        if not self.stop_event.is_set():
            self.stop_event.set()
