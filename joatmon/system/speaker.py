import math
import queue
import sys
import threading
import time
import warnings

from joatmon.system.core import RWLock

MSSAM = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSSam'
MSMARY = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSMary'
MSMIKE = 'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSMike'

E_REG = {
    MSSAM: (137.89, 1.11),
    MSMARY: (156.63, 1.11),
    MSMIKE: (154.37, 1.11)
}

if sys.platform != 'win32':
    warnings.warn('The speaker module can only be used on a Windows system. TTS will never be enabled')


def to_utf8(value):
    return str(value).encode('utf-8')


def from_utf8(value):
    return value.decode('utf-8')


class Voice(object):
    def __init__(self, voice_id, name=None, languages=None, gender=None, age=None):
        self.id = voice_id
        self.name = name
        self.languages = languages
        self.gender = gender
        self.age = age

    def __str__(self):
        return """<Voice id=%(id)s name=%(name)s languages=%(languages)s gender=%(gender)s age=%(age)s>""" % self.__dict__


class OutputDevice:
    def __init__(self, tts_enabled):
        super(OutputDevice, self).__init__()

        self.lock = RWLock()
        self.speaking_event = threading.Event()
        self.stop_event = threading.Event()
        self.events = queue.Queue()

        self.tts_enabled = tts_enabled and sys.platform == 'win32'
        self.current_speaker = None

        self.rate_wpm = 200

    def output(self, text):
        if not text.endswith('\n'):
            text += '\n'
        self.write(text)
        self.flush()

    def write(self, text):
        with self.lock.w_locked():
            if self.tts_enabled:
                self.say(text)
            else:
                sys.stdout.write(text)

    def flush(self):
        # when flushed, might want to need to wait until the speaking is over
        # might want to use rwlock as well
        if not self.tts_enabled:
            sys.stdout.flush()

    def say(self, text):
        # speak(x, 1) # separate thread
        # speak(x) # current thread
        # speak('', 3) # stop talking
        # self._tts.Speak(from_utf8(to_utf8(text)), 19)
        import pythoncom
        import win32com.client

        def thread_function():
            class SpeakerEvents(object):
                @staticmethod
                def OnStartStream(*args):
                    ...

                @staticmethod
                def OnEndStream(*args):
                    self.speaking_event.clear()

                @staticmethod
                def OnVoiceChange(*args):
                    ...

            self.speaking_event.set()

            speaker = win32com.client.Dispatch("SAPI.SpVoice", pythoncom.CoInitialize())
            self.current_speaker = speaker
            speaker.EventInterests = 33790

            self.set_property(speaker, 'voice', self.get_property(speaker, 'voice'))

            win32com.client.WithEvents(speaker, SpeakerEvents)

            speaker.Speak(from_utf8(to_utf8(text)), 19)

            while self.speaking_event.is_set():
                pythoncom.PumpWaitingMessages()

        if self.speaking_event.is_set():
            if self.current_speaker is not None:
                self.current_speaker.Speak('', 3)
                self.current_speaker = None
            self.speaking_event.clear()
            time.sleep(0.4)

        thread = threading.Thread(target=thread_function, args=())
        thread.start()

    def stop(self):
        if self.speaking_event.is_set():
            if self.current_speaker is not None:
                self.current_speaker.Speak('', 3)
                self.current_speaker = None
            self.speaking_event.clear()
            time.sleep(0.4)
        self.stop_event.set()

    def _token_from_id(self, speaker, id_):
        tokens = speaker.GetVoices()
        for token in tokens:
            if token.Id == id_:
                return token
        for t in speaker.GetVoices():
            return t
        raise ValueError('unknown voice id %s', id_)

    def get_property(self, speaker, name):
        if name == 'voices':
            return [Voice(attr.Id, attr.GetDescription()) for attr in speaker.GetVoices()]
        elif name == 'voice':
            return speaker.Voice.Id
        elif name == 'rate':
            return self.rate_wpm
        elif name == 'volume':
            return speaker.Volume / 100.0
        else:
            raise KeyError('unknown property %s' % name)

    def set_property(self, speaker, name, value):
        if name == 'voice':
            token = self._token_from_id(speaker, value)
            speaker.Voice = token
            a, b = E_REG.get(value, E_REG[MSMARY])
            speaker.Rate = int(math.log(self.rate_wpm / a, b))
        elif name == 'rate':
            id_ = speaker.Voice.Id
            a, b = E_REG.get(id_, E_REG[MSMARY])
            try:
                speaker.Rate = int(math.log(value / a, b))
            except TypeError as e:
                raise ValueError(str(e))
            self.rate_wpm = value
        elif name == 'volume':
            try:
                speaker.Volume = int(round(value * 100, 2))
            except TypeError as e:
                raise ValueError(str(e))
        else:
            raise KeyError('unknown property %s' % name)
