import json
import os

from elevenlabs import (
    generate,
    set_api_key
)


class TTSAgent:
    def __init__(self):
        super(TTSAgent, self).__init__()

        set_api_key(json.loads(open('iva/iva.json', 'r').read())['config']['elevenlabs']['key'])

    def convert(self, text):
        text_hash = ''.join([c.lower() for c in text.lower().strip() if c not in ['!', '?', ',', '/', '\\']]).replace(' ', '_').replace('.', '_').replace(':', '_')
        text_audio_path = os.path.join('iva/cache', f'{text_hash}.mp3')

        if not os.path.exists(os.path.dirname(text_audio_path)):
            os.makedirs(os.path.dirname(text_audio_path))

        if os.path.exists(text_audio_path):
            audio = open(text_audio_path, 'rb').read()
        else:
            audio = generate(
                text=text,
                voice=json.loads(open('iva/iva.json', 'r').read())['config']['elevenlabs']['voice'],
                model=json.loads(open('iva/iva.json', 'r').read())['config']['elevenlabs']['model']
            )
            open(text_audio_path, 'wb').write(audio)

        return audio
