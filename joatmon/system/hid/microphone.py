import queue
import threading

import speech_recognition as sr


class Microphone:
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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return input()

        # return self.audio_queue.get()

    def stop(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.stop_event.set()
