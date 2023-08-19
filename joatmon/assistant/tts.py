import json
import os

from elevenlabs import (
    generate,
    set_api_key
)


class TTSAgent:
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
        super(TTSAgent, self).__init__()

        set_api_key(json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['elevenlabs']['key'])

    def convert(self, text):
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
            audio = generate(
                text=text,
                voice=json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['elevenlabs']['voice'],
                model=json.loads(open(os.path.join(os.environ.get('IVA_PATH'), 'iva.json'), 'r').read())['config']['elevenlabs']['model'],
            )
            open(text_audio_path, 'wb').write(audio)

        return audio
