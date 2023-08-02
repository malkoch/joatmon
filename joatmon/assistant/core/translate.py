from __future__ import print_function

import json

import openai

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api, **kwargs):
        super(Task, self).__init__(api, **kwargs)

    @staticmethod
    def help():
        return {
            "name": "translate",
            "description": "a function for user to translate a message",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "message to be translated"
                    }
                },
                "required": ["message"]
            }
        }

    def run(self):
        config = json.loads(open('iva/iva.json', 'r').read())['config']['openai']
        openai.api_key = config['key']

        message = self.kwargs.get('message', None) or self.api.input('what do you want the message to be')

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": 'from now on you are going to translate my input and generate only the translated message'},
                {
                    "role": "agent", "content": 'Understood! From now on, I will translate your input and provide the translated message. '
                                                'Please go ahead and let me know what you '
                                                'would like to be translated.'
                },
                {"role": "user", "content": message}
            ]
        )
        self.api.output(response['choices'][0]['message']['content'])

        if not self.stop_event.is_set():
            self.stop_event.set()
