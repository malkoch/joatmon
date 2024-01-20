#


## Runnable
```python 
Runnable(
   info, api, type, **kwargs
)
```


---
Runnable class for managing asynchronous tasks.

This class provides a way to manage asynchronous tasks, including starting, stopping, and checking the running status of the tasks.


**Attributes**

* **machine** (Machine) : The state machine for managing the state of the task.
* **process_id** (uuid.UUID) : The unique identifier for the task.
* **info** (str) : The information about the task.
* **api** (object) : The API object.
* **type** (str) : The type of the task.
* **kwargs** (dict) : Additional keyword arguments.
* **event** (asyncio.Event) : An event for signaling the termination of the task.
* **task** (asyncio.Task) : The task that is being run.


**Args**

* **info** (str) : The information about the task.
* **api** (object) : The API object.
* **type** (str) : The type of the task.
* **kwargs** (dict) : Additional keyword arguments.



**Methods:**


### .help
```python
.help()
```

---
Provide help about the Runnable.


**Returns**

* **dict**  : An empty dictionary as this method needs to be implemented in subclasses.


### .run
```python
.run()
```

---
Run the task.

This method needs to be implemented in subclasses.

### .running
```python
.running()
```

---
Check if the task is running.


**Returns**

* **bool**  : True if the task is running, False otherwise.


### .start
```python
.start()
```

---
Start the task.

This method starts the task by setting the state to 'starting', firing the 'begin' event, and then setting the state to 'started'.

### .stop
```python
.stop()
```

---
Stop the task.

This method stops the task by setting the state to 'stopping', firing the 'end' event, setting the event, setting the state to 'stopped', and then cancelling the task.
