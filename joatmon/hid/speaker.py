import sys
import threading

from joatmon.system.core import RWLock


class OutputDevice:
    def __init__(self, tts_enabled):
        super(OutputDevice, self).__init__()

        self._tts = tts_enabled

        self.stop_event = threading.Event()

        self.lock = RWLock()

    @property
    def tts_enabled(self):
        return self._tts

    @tts_enabled.setter
    def tts_enabled(self, value):
        self._tts = value

    def output(self, text):
        with self.lock.w_locked():
            if not text.endswith('\n'):
                text += '\n'
            self.write(text)
            self.flush()

    def write(self, text):
        if self.tts_enabled:
            self.say(text)
        else:
            sys.stdout.write(text)

    def flush(self):
        if not self.tts_enabled:
            sys.stdout.flush()

    def say(self, text):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def stop(self):
        self.stop_event.set()
