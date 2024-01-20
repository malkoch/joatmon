import threading

import cv2


class Camera:
    """
    Camera class that provides the functionality for capturing video from a camera.

    Attributes:
        cam (cv2.VideoCapture): The camera object.
        stop_event (threading.Event): The event to stop the camera.

    Methods:
        frame: Captures a single frame from the camera.
        shot: Captures a single frame from the camera and saves it to a file.
        stream: Streams video from the camera.
        record: Records video from the camera for a specified amount of time.
        stop: Stops the camera.
    """

    def __init__(self):
        """
        Initialize Camera.
        """
        super(Camera, self).__init__()

        self.cam = None
        self.stop_event = threading.Event()

    def frame(self):
        """
        Captures a single frame from the camera.

        Returns:
            numpy.ndarray: The captured frame.
        """
        self.cam = cv2.VideoCapture(0)
        ret, frame = self.cam.read()

        return frame

    def shot(self, path):
        """
        Captures a single frame from the camera and saves it to a file.

        Args:
            path (str): The path to save the frame.
        """

    def stream(self):
        """
        Streams video from the camera.

        Yields:
            numpy.ndarray: The current frame.
        """
        self.cam = cv2.VideoCapture(0)

        while not self.stop_event.is_set():
            ret, frame = self.cam.read()
            yield frame

    def record(self, path, time):
        """
        Records video from the camera for a specified amount of time.

        Args:
            path (str): The path to save the video.
            time (int): The amount of time to record video.
        """

    def stop(self):
        """
        Stops the camera.
        """
        self.stop_event.set()
        if self.cam is not None:
            self.cam.release()
