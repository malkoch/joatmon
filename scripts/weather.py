from __future__ import print_function

import json
from datetime import datetime

import requests

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    run_arguments = {
        'functionality': '',
        'location': ''
    }

    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['weather']
        base_url = config['url']
        key = config['key']
        if self.kwargs['functionality'] == 'current':
            resp = requests.get(base_url + 'current.json', params={'key': key, 'q': self.kwargs['location']})

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

            self.api.output(
                f'Current Temperature is: {temp_c} and it feels like {feelslike_c}, '
                f'Current Humidity is: {humidity}'
            )
        elif self.kwargs['functionality'] == 'forecast':
            resp = requests.get(base_url + 'forecast.json', params={'key': key, 'q': self.kwargs['location'], 'days': 3})
        elif self.kwargs['functionality'] == 'search':
            resp = requests.get(base_url + 'search.json', params={'key': key, 'q': self.kwargs['location']})
        elif self.kwargs['functionality'] == 'history':
            resp = requests.get(base_url + 'history.json', params={'key': key, 'q': self.kwargs['location'], 'dt': datetime.now().isoformat()})
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()


class Job(BaseTask):
    arguments = {
        'functionality': '',
        'location': ''
    }

    def __init__(self, api, **kwargs):
        super(Job, self).__init__(api, **kwargs)

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['weather']
        base_url = config['url']
        key = config['key']
        if self.kwargs['functionality'] == 'current':
            resp = requests.get(base_url + 'current.json', params={'key': key, 'q': self.kwargs['location']})

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
        elif self.kwargs['functionality'] == 'forecast':
            resp = requests.get(base_url + 'forecast.json', params={'key': key, 'q': self.kwargs['location'], 'days': 3})
        elif self.kwargs['functionality'] == 'search':
            resp = requests.get(base_url + 'search.json', params={'key': key, 'q': self.kwargs['location']})
        elif self.kwargs['functionality'] == 'history':
            resp = requests.get(base_url + 'history.json', params={'key': key, 'q': self.kwargs['location'], 'dt': datetime.now().isoformat()})
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
