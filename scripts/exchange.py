from __future__ import print_function

import json

import requests

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def params():
        return []

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['exchange']
        base_url = config['url']
        key = config['key']

        resp = requests.get(base_url + 'exchange_rate', params={'apikey': key, 'symbol': 'USD/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        self.api.say(f'{response["symbol"]} : {response["rate"]}')

        resp = requests.get(base_url + 'exchange_rate', params={'apikey': key, 'symbol': 'EUR/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        self.api.say(f'{response["symbol"]} : {response["rate"]}')
