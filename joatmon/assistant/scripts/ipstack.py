from __future__ import print_function

import argparse
import json
import sys

import requests

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--args', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = [namespace.args]

    def run(self):
        print(f'dummy task is running {self.action}')
        config = json.loads(open('iva.json', 'r').read())['configs']['arxiv']

        send_url = f"{config['url']}/check?access_key={config['token']}"
        geo_req = requests.get(send_url)
        geo_json = json.loads(geo_req.text)
        latitude = geo_json['latitude']
        latitude = str(latitude)
        longitude = geo_json['longitude']
        longitude = str(longitude)
        city = geo_json['city']
        continent_name = geo_json['continent_name']
        country_name = geo_json['country_name']
        pin = geo_json['zip']
        pin = str(pin)

        if not self.event.is_set():
            self.event.set()

        return {
            'longitude': longitude,
            'latitude': latitude,
            'city': city,
            'country': country_name,
            'continent': continent_name,
            'pin': pin
        }


if __name__ == '__main__':
    Task(None).run()
