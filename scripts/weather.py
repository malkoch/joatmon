from __future__ import print_function

import json
from datetime import datetime

import requests

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def params():
        return ['functionality', 'location'] + BaseTask.params()

    def run(self):
        functionality = self.kwargs.get('functionality', '') or self.api.listen('what do you want the functionality to be')
        location = self.kwargs.get('location', '') or self.api.listen('what do you want the location to be')

        config = json.loads(open('iva.json', 'r').read())['configs']['weather']
        base_url = config['url']
        key = config['key']
        if functionality == 'current':
            resp = requests.get(base_url + 'current.json', params={'key': key, 'q': location})

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

            todo = self.kwargs.get('todo', '') or self.api.listen('what do you want me to do with it')
            self.api.run_task(todo, {'message': f'Current Temperature is: {temp_c} and it feels like {feelslike_c}, Current Humidity is: {humidity}'})
        elif functionality == 'forecast':
            resp = requests.get(base_url + 'forecast.json', params={'key': key, 'q': self.kwargs['location'], 'days': 3})
        elif functionality == 'search':
            resp = requests.get(base_url + 'search.json', params={'key': key, 'q': self.kwargs['location']})
        elif functionality == 'history':
            resp = requests.get(base_url + 'history.json', params={'key': key, 'q': self.kwargs['location'], 'dt': datetime.now().isoformat()})
        else:
            raise ValueError(f'arguments are not recognized')

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
