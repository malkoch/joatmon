import queue
import threading

import speech_recognition as sr


class Microphone:
    def __init__(self):
        super(Microphone, self).__init__()

        self.r = sr.Recognizer()
        self.r.energy_threshold = 350
        self.r.pause_threshold = 0.8
        self.r.dynamic_energy_threshold = False
        self.r.timeout = 1

        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.stop_event = threading.Event()

        self.listening_thread = threading.Thread(target=self.record_audio)

        # self.listening_thread.start()

    def record_audio(self):
        with sr.Microphone(sample_rate=16000) as source:
            # self.r.adjust_for_ambient_noise(source)
            while not self.stop_event.is_set():
                try:
                    audio = self.r.listen(source, timeout=1)
                    self.audio_queue.put_nowait(audio)
                except sr.exceptions.WaitTimeoutError:
                    ...

    def listen(self):
        return input()

        # return self.audio_queue.get()

    def stop(self):
        self.stop_event.set()
