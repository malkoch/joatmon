import time

from joatmon.system.microphone import InputDriver
from joatmon.system.speaker import OutputDevice

device = OutputDevice(True)

driver = InputDriver(device, True)

while True:
    time.sleep(0.1)
