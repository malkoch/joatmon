import subprocess

from joatmon.system import is_installed


class Speaker:
    """
    A class used to represent a Speaker.

    ...

    Methods
    -------
    __init__(self)
        Initializes a new instance of the Speaker class.
    say(self, audio)
        Plays the specified audio.
    """

    def __init__(self):
        """
        Initializes a new instance of the Speaker class.
        """
        super(Speaker, self).__init__()

    def say(self, audio):
        """
        Plays the specified audio.

        Args:
            audio (bytes): The audio to be played.
        """
        play(audio)


def play(audio: bytes) -> None:
    """
    Plays the specified audio using ffplay from ffmpeg.

    Args:
        audio (bytes): The audio to be played.

    Raises:
        ValueError: If ffplay from ffmpeg is not installed.
    """
    if not is_installed("ffplay"):
        raise ValueError("ffplay from ffmpeg not found, necessary to play audio.")
    args = ["ffplay", "-autoexit", "-", "-nodisp"]
    proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(input=audio)
    proc.poll()
