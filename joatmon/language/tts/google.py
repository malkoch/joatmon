import base64
import json
import os

import requests

from joatmon.language.tts.core import TTSAgent


class GoogleTTS(TTSAgent):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, config):
        super(TTSAgent, self).__init__()

        self._config = config

    def convert(self, text):  # cachable
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

        text_hash = (
            ''.join([c.lower() for c in text.lower().strip() if c not in ['!', '?', ',', '/', '\\']])
            .replace(' ', '_')
            .replace('.', '_')
            .replace(':', '_')
        )
        text_audio_path = os.path.join(os.environ.get('IVA_PATH'), 'cache', f'{text_hash}.mp3')

        if not os.path.exists(os.path.dirname(text_audio_path)):
            os.makedirs(os.path.dirname(text_audio_path))

        if os.path.exists(text_audio_path):
            audio = open(text_audio_path, 'rb').read()
        else:
            response = requests.post(
                f'https://texttospeech.googleapis.com/v1/text:synthesize?alt=json&key={self._config["key"]}', data=json.dumps(
                    {
                        'input': {
                            'text': text
                        },
                        'voice': {
                            'languageCode': self._config["language"],
                            'name': self._config["voice"],
                            # ssmlGender
                            # customVoice
                        },
                        'audioConfig': {
                            'audioEncoding': 1,
                            'speakingRate': self._config["speed"],
                            'pitch': self._config["pitch"],
                            'volumeGainDb': self._config["volume"],
                            # sampleRateHertz
                            # effectsProfileId
                        }
                    }
                )
            )
            base64_audio = json.loads(response.content.decode('utf-8'))['audioContent']
            audio = base64.b64decode(base64_audio)
            open(text_audio_path, 'wb').write(audio)

        return audio
