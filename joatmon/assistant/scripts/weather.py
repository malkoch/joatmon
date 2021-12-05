from __future__ import print_function

import argparse
import json
import sys
from datetime import datetime

import requests

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api=None):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--current', type=str)
        parser.add_argument('--forecast', type=str)
        parser.add_argument('--search', type=str)
        parser.add_argument('--history', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.current:
            self.action = ['current', namespace.current]
        elif namespace.forecast:
            self.action = ['download', namespace.forecast]
        elif namespace.search:
            self.action = ['search', namespace.search]
        elif namespace.history:
            self.action = ['history', namespace.history]

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['weather']
        base_url = config['url']
        headers = {
            'x-rapidapi-host': config['host'],
            'x-rapidapi-key': config['key']
        }
        if self.action[0] == 'current':
            resp = requests.get(base_url + 'current.json', params={'key': config['key'], 'q': self.action[1]}, headers=headers)

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

            if self.api is not None:
                self.api.output(
                    f'Current Temperature is: {temp_c} and it feels like {feelslike_c}, '
                    f'Current Humidity is: {humidity}'
                )
            else:
                print(
                    f'Current Temperature is: {temp_c} and it feels like {feelslike_c}, '
                    f'Current Humidity is: {humidity}'
                )
        elif self.action[0] == 'forecast':
            resp = requests.get(base_url + 'forecast.json', params={'key': config, 'q': self.action[1], 'days': 3}, headers=headers)
        elif self.action[0] == 'search':
            resp = requests.get(base_url + 'search.json', params={'key': config, 'q': self.action[1]}, headers=headers)
        elif self.action[0] == 'history':
            resp = requests.get(base_url + 'history.json', params={'key': config, 'q': self.action[1], 'dt': datetime.now().isoformat()}, headers=headers)
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
