from __future__ import print_function

import json

import openai

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return ''

    @staticmethod
    def params():
        return []

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['openai']
        openai.api_key = config['key']

        message = self.api.listen('what do you want the message to be')

        history = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": message}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=history
        )
        response = response['choices'][0]['message']['content']

        todo = self.kwargs.get('todo', '') or self.api.listen('what do you want me to do with it')
        self.api.run_task(todo, {'message': response, 'path': r'C:\Users\malkoch\Documents\Github\joatmon\scripts\a.py'})

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
