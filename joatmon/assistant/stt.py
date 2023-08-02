import json
from io import BytesIO

import openai


class STTAgent:
    def __init__(self):
        super(STTAgent, self).__init__()

        openai.api_key = json.loads(open('iva/iva.json', 'r').read())['config']['openai']['key']

    def transcribe(self, audio):
        wav_data = BytesIO(audio.get_wav_data())
        wav_data.name = "SpeechRecognition_audio.wav"

        transcript = openai.Audio.transcribe('whisper-1', wav_data)
        return transcript["text"]

        # ignore = ['!', '?', ',', '.']
        # result = self.result_queue.get()
        # return ''.join([c.lower() for c in result.lower().strip() if c not in ignore])
