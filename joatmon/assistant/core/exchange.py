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
            "name": "exchange",
            "description": "a function for user to learn exchange rates compared to TRY",
            "parameters": {
                "type": "object",
                "properties": {
                },
                "required": []
            }
        }

    def run(self):
        key = json.loads(open('iva.json', 'r').read())['config']['exchange']['key']

        data = []

        resp = requests.get('https://api.twelvedata.com/exchange_rate', params={'apikey': key, 'symbol': 'USD/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        data.append(f'{response["symbol"]} : {response["rate"]}')

        resp = requests.get('https://api.twelvedata.com/exchange_rate', params={'apikey': key, 'symbol': 'EUR/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        data.append(f'{response["symbol"]} : {response["rate"]}')

        self.api.say('\n'.join(data))

        if not self.stop_event.is_set():
            self.stop_event.set()
