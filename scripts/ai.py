from __future__ import print_function

import argparse
import json
import sys
import time

import openai

from joatmon.assistant.task import BaseTask


class Task(BaseTask):
    def __init__(self, api=None):
        super(Task, self).__init__(api, False, 1, 100)

        parser = argparse.ArgumentParser()
        parser.add_argument('--ask', type=str)
        parser.add_argument('--image', type=str)
        parser.add_argument('--transcribe', type=str)

        namespace, _ = parser.parse_known_args(sys.argv)

        self.action = None
        if namespace.ask:
            self.action = ['ask', namespace.ask]
        elif namespace.image:
            self.action = ['image', namespace.image]
        elif namespace.transcribe:
            self.action = ['transcribe', namespace.transcribe]

    @staticmethod
    def help(api):
        message = """
        this module can be used to use openai api
            --ask to use chat-gpt
            --image to use dall-e
            --transcribe to use whisper
        """
        if api is not None:
            api.output(message)
            time.sleep(7)
        else:
            print(message)

    def run(self):
        try:
            config = json.loads(open('iva.json', 'r').read())['configs']['openai']
            openai.api_key = config['key']

            if self.action[0] == 'ask':
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": self.action[1]}]
                )
                print(response['choices'][0]['message']['content'])
            elif self.action[0] == 'image':
                response = openai.Image.create(
                    prompt=self.action[1],
                    n=1,
                    size="1024x1024"
                )
                print(response)
            elif self.action[0] == 'transcribe':
                audio_file = open("/path/to/file/audio.mp3", "rb")
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                print(transcript)
            else:
                raise ValueError(f'arguments are not recognized')
        except Exception as ex:
            print(str(ex))

        if not self.event.is_set():
            self.event.set()


if __name__ == '__main__':
    Task(None).run()
