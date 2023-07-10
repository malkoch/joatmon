import json
import os

from elevenlabs import (
    generate,
    play,
    set_api_key
)


class OutputDevice:
    def __init__(self):
        super(OutputDevice, self).__init__()

        set_api_key(json.loads(open('iva.json', 'r').read())['config']['elevenlabs']['key'])

    def say(self, text):
        text_hash = ''.join([c.lower() for c in text.lower().strip() if c not in ['!', '?', ',', '/', '\\']]).replace(' ', '_').replace('.', '_').replace(':', '_')
        text_audio_path = os.path.join('cache', f'{text_hash}.mp3')

        if not os.path.exists(os.path.dirname(text_audio_path)):
            os.makedirs(os.path.dirname(text_audio_path))

        if os.path.exists(text_audio_path):
            audio = open(text_audio_path, 'rb').read()
            play(audio)
        else:
            try:
                audio = generate(
                    text=text,
                    voice=json.loads(open('iva.json', 'r').read())['config']['elevenlabs']['voice'],
                    model=json.loads(open('iva.json', 'r').read())['config']['elevenlabs']['model']
                )
                open(text_audio_path, 'wb').write(audio)
                play(audio)
            except Exception as ex:
                print(str(ex))
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 175)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
