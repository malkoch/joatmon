#


## Speaker
```python 

```


---
A class used to represent a Speaker.

...

Methods
-------
__init__(self)
Initializes a new instance of the Speaker class.
---
    Plays the specified audio.


**Methods:**


### .say
```python
.say(
   audio
)
```

---
Plays the specified audio.


**Args**

* **audio** (bytes) : The audio to be played.


----


### play
```python
.play(
   audio: bytes
)
```

---
Plays the specified audio using ffplay from ffmpeg.


**Args**

* **audio** (bytes) : The audio to be played.


**Raises**

* **ValueError**  : If ffplay from ffmpeg is not installed.

