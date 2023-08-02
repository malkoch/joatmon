from __future__ import print_function

import json

import requests

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "weather",
            "description": "a function for user to learn current, historical, forecast weather data for given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["current", "history", "forecast"]
                    },
                    "location": {
                        "type": "string",
                        "description": "location of the weather"
                    }
                },
                "required": ["mode", "location"]
            }
        }

    def run(self):
        key = json.loads(open('iva/iva.json', 'r').read())['config']['weather']['key']

        mode = self.kwargs.get('mode', None)
        location = self.kwargs.get('location', None) or self.api.input('what is the location')

        if mode == 'current':
            resp = requests.get('https://api.weatherapi.com/v1/current.json', params={'key': key, 'q': location})

            response = json.loads(resp.content.decode('utf-8'))
            current = response.get('current', {})
            temp_c = current.get('temp_c', None)
            condition = current.get('condition', {}).get('text', None)
            wind_kph = current.get('wind_kph', None)
            wind_degree = current.get('wind_degree', None)
            wind_dir = current.get('wind_dir', None)
            pressure_mb = current.get('pressure_mb', None)
            humidity = current.get('humidity', None)
            cloud = current.get('cloud', None)
            feelslike_c = current.get('feelslike_c', None)
            vis_km = current.get('vis_km', None)

            self.api.output(f'Current Temperature is: {temp_c} and it feels like {feelslike_c}, Current Humidity is: {humidity}')

        if not self.stop_event.is_set():
            self.stop_event.set()
