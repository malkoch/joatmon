class OutputDevice:
    def __init__(self):
        super(OutputDevice, self).__init__()

    def say(self, text):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
