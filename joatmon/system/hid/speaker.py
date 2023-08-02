from elevenlabs import play


class Speaker:
    def __init__(self):
        super(Speaker, self).__init__()

        # set_api_key(json.loads(open('iva/iva.json', 'r').read())['config']['elevenlabs']['key'])

    def say(self, audio):
        play(audio)
