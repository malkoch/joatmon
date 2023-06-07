import queue
import threading


class InputDriver:
    def __init__(self):
        super(InputDriver, self).__init__()

        import whisper
        self.audio_model = whisper.load_model('small.en').cuda()
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.stop_event = threading.Event()

        self.listening_thread = threading.Thread(target=self.record_audio)
        self.translator_thread = threading.Thread(target=self.transcribe_forever)

        self.listening_thread.start()
        self.translator_thread.start()

    def record_audio(self):
        import speech_recognition as sr
        import speech_recognition.exceptions

        r = sr.Recognizer()
        r.energy_threshold = 100
        r.pause_threshold = 0.8
        r.dynamic_energy_threshold = True
        r.timeout = 1

        with sr.Microphone(sample_rate=16000) as source:
            import numpy as np
            import torch

            while not self.stop_event.is_set():
                try:
                    audio = r.listen(source, timeout=1)
                    audio_data = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)

                    self.audio_queue.put_nowait(audio_data)
                except speech_recognition.exceptions.WaitTimeoutError:
                    ...

    def transcribe_forever(self):
        while not self.stop_event.is_set():
            try:
                audio_data = self.audio_queue.get_nowait().cuda()
                result = self.audio_model.transcribe(audio_data, language='english')
                self.result_queue.put_nowait(result)
            except queue.Empty:
                ...

    def listen(self):
        # need to find a way to remove these and keep punctuations as well
        return self.result_queue.get()['text'].lower().strip()

    def stop(self):
        self.stop_event.set()
