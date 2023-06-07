from __future__ import print_function

import json

import openai

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def create(api):
        api.output('what do you want the message to be')
        message = api.input()
        return {'message': message}

    def run(self):
        config = json.loads(open('iva.json', 'r').read())['configs']['openai']
        openai.api_key = config['key']

        if not (message := self.kwargs.get('message', '')):
            self.api.output('what do you want the message to be')
            message = self.api.input()

        history = [{"role": "system", "content": "You are a helpful assistant."}]

        while True:
            history.append({"role": "user", "content": message})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=history
            )
            response = response['choices'][0]['message']['content']
            history.append({"role": "assistant", "content": response})
            self.api.output(response)

            self.api.output('what do you want the message to be')
            message = self.api.input()
            if message == '':
                break

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
