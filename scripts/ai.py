from __future__ import print_function

import json

import openai

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    run_arguments = {
        'message': ''
    }

    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['openai']
        openai.api_key = config['key']

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": self.kwargs['message']}]
        )
        self.api.output(response['choices'][0]['message']['content'])

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
