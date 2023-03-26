import sys
import threading
from io import TextIOBase


class InputDriver(TextIOBase):
    def __init__(self, stt_enabled):
        super(InputDriver, self).__init__()

        self._stt = stt_enabled

        if self.stt_enabled:
            import queue
            import whisper

            self.audio_model = whisper.load_model('small.en').cuda()
            self.audio_queue = queue.Queue()
            self.result_queue = queue.Queue()

            self.listening_thread = threading.Thread(target=self.record_audio)
            self.translator_thread = threading.Thread(target=self.transcribe_forever)

            self.listening_thread.start()
            self.translator_thread.start()

        self.stop_event = threading.Event()

    @property
    def stt_enabled(self):
        return self._stt

    @stt_enabled.setter
    def stt_enabled(self, value):
        self._stt = value

    def record_audio(self):
        import speech_recognition as sr

        r = sr.Recognizer()
        r.energy_threshold = 100
        r.pause_threshold = 0.8
        r.dynamic_energy_threshold = True

        with sr.Microphone(sample_rate=16000) as source:
            import numpy as np
            import torch
            print("Say something!")
            while not self.stop_event.is_set():
                audio = r.listen(source)
                torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
                audio_data = torch_audio

                self.audio_queue.put_nowait(audio_data)

    def transcribe_forever(self):
        while not self.stop_event.is_set():
            audio_data = self.audio_queue.get().cuda()
            result = self.audio_model.transcribe(audio_data, language='english')
            self.result_queue.put_nowait(result)

    def readline(self, **kwargs):
        if not self.stt_enabled:
            return sys.stdin.readline()
        else:
            return self.result_queue.get()['text'].replace('.', '').lower()

    def stop(self):
        self.stop_event.set()
