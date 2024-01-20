#


## Microphone
```python 

```


---
A class used to represent a Microphone.

...

Attributes
----------
r : sr.Recognizer
The recognizer instance used to recognize speech.
---
    The queue to store audio data.
    The queue to store the result of speech recognition.
    The event to signal the stop of the listening thread.
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


**Methods:**


### .record_audio
```python
.record_audio()
```

---
Records audio from the microphone and puts it into the audio queue.

### .listen
```python
.listen()
```

---
Gets the next audio data from the audio queue.


**Returns**

* **AudioData**  : The next audio data from the audio queue.


### .stop
```python
.stop()
```

---
Stops the listening thread.
