import queue
import threading

import speech_recognition as sr


class Microphone:
    """
    A class used to represent a Microphone.

    ...

    Attributes
    ----------
    r : sr.Recognizer
        The recognizer instance used to recognize speech.
    audio_queue : queue.Queue
        The queue to store audio data.
    result_queue : queue.Queue
        The queue to store the result of speech recognition.
    stop_event : threading.Event
        The event to signal the stop of the listening thread.
    listening_thread : threading.Thread
        The thread to listen to the microphone.

    Methods
    -------
    __init__(self)
        Initializes a new instance of the Microphone class.
    record_audio(self)
        Records audio from the microphone and puts it into the audio queue.
    listen(self)
        Gets the next audio data from the audio queue.
    stop(self)
        Stops the listening thread.
    """

    def __init__(self):
        """
        Initializes a new instance of the Microphone class.
        """
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

        self.listening_thread.start()

    def record_audio(self):
        """
        Records audio from the microphone and puts it into the audio queue.
        """
        with sr.Microphone(sample_rate=16000) as source:
            # self.r.adjust_for_ambient_noise(source)
            while not self.stop_event.is_set():
                try:
                    audio = self.r.listen(source, timeout=1)
                    self.audio_queue.put_nowait(audio)
                except sr.exceptions.WaitTimeoutError:
                    ...

    def listen(self):
        """
        Gets the next audio data from the audio queue.

        Returns:
            sr.AudioData: The next audio data from the audio queue.
        """
        return self.audio_queue.get()

    def stop(self):
        """
        Stops the listening thread.
        """
        self.stop_event.set()
