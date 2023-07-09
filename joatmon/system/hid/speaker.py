import json

from elevenlabs import (
    generate,
    set_api_key,
    stream
)


class OutputDevice:
    def __init__(self):
        super(OutputDevice, self).__init__()

        set_api_key(json.loads(open('iva.json', 'r').read())['config']['elevenlabs']['key'])

    def say(self, text):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

        # audio_stream = generate(
        #     text=text,
        #     voice=json.loads(open('iva.json', 'r').read())['config']['elevenlabs']['voice'],
        #     model=json.loads(open('iva.json', 'r').read())['config']['elevenlabs']['model'],
        #     stream=True
        # )
#
        # stream(audio_stream)
