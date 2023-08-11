import threading

import cv2


class Camera:
    def __init__(self):
        super(Camera, self).__init__()

        self.cam = None
        self.stop_event = threading.Event()

    def frame(self):
        self.cam = cv2.VideoCapture(0)
        ret, frame = self.cam.read()

        return frame

    def shot(self, path):
        ...

    def stream(self):
        self.cam = cv2.VideoCapture(0)

        while not self.stop_event.is_set():
            ret, frame = self.cam.read()
            yield frame

    def record(self, path, time):
        ...

    def stop(self):
        self.stop_event.set()
        if self.cam is not None:
            self.cam.release()
