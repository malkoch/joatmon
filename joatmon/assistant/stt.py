import json
from io import BytesIO

import openai


class STTAgent:
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

    def __init__(self):
        super(STTAgent, self).__init__()

        openai.api_key = json.loads(open('iva/iva.json', 'r').read())['config']['openai']['key']

    def transcribe(self, audio):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        wav_data = BytesIO(audio.get_wav_data())
        wav_data.name = 'SpeechRecognition_audio.wav'

        transcript = openai.Audio.transcribe('whisper-1', wav_data)
        return transcript['text']

        # ignore = ['!', '?', ',', '.']
        # result = self.result_queue.get()
        # return ''.join([c.lower() for c in result.lower().strip() if c not in ignore])
