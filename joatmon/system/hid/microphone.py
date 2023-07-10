import json
import queue
import threading

import openai
import speech_recognition as sr


class InputDriver:
    def __init__(self, o):
        super(InputDriver, self).__init__()

        self.o = o

        openai.api_key = json.loads(open('iva.json', 'r').read())['config']['openai']['key']

        self.r = sr.Recognizer()
        self.r.energy_threshold = 350
        self.r.pause_threshold = 0.8
        self.r.dynamic_energy_threshold = False
        self.r.timeout = 1

        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.stop_event = threading.Event()

        self.listening_thread = threading.Thread(target=self.record_audio)
        self.translator_thread = threading.Thread(target=self.transcribe_forever)

        # self.listening_thread.start()
        # self.translator_thread.start()

    def record_audio(self):
        with sr.Microphone(sample_rate=16000) as source:
            # self.r.adjust_for_ambient_noise(source)
            while not self.stop_event.is_set():
                try:
                    audio = self.r.listen(source, timeout=1)
                    self.audio_queue.put_nowait(audio)
                except sr.exceptions.WaitTimeoutError:
                    ...

    def transcribe_forever(self):
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get_nowait()
                result = self.r.recognize_whisper_api(audio, model='whisper-1', api_key=json.loads(open('iva.json', 'r').read())['config']['openai']['key'])
                self.result_queue.put_nowait(result)
            except queue.Empty:
                ...

    def listen(self, prompt=None):
        if prompt is not None:
            self.o.say(prompt)

        prompt = prompt or 'prompt'
        return input(f'{prompt}: ')

        # ignore = ['!', '?', ',', '.']
        # result = self.result_queue.get()
        # return ''.join([c.lower() for c in result.lower().strip() if c not in ignore])

    def stop(self):
        self.stop_event.set()
