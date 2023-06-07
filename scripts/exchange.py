from __future__ import print_function

import json
from datetime import datetime

import requests

from joatmon.assistant.task import BaseTask


class Job(BaseTask):
    def __init__(self, api, **kwargs):
        super(Job, self).__init__(api, **kwargs)

    @staticmethod
    def create(api):
        return {}

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['exchange']
        base_url = config['url']
        key = config['key']

        ui = []

        resp = requests.get(base_url + 'exchange_rate', params={'apikey': key, 'symbol': 'USD/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        ui.append(f'{response["symbol"]} : {response["rate"]}')

        resp = requests.get(base_url + 'exchange_rate', params={'apikey': key, 'symbol': 'EUR/TRY'})
        response = json.loads(resp.content.decode('utf-8'))
        ui.append(f'{response["symbol"]} : {response["rate"]}')

        self.api.show_(
            'l2',
            'exchange',
            ui
        )
