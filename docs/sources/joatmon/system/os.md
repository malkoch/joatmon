#


## Task
```python 
Task()
```



----


## Service
```python 
Service()
```



----


## State
```python 
State()
```



----


## Result
```python 
Result()
```



----


## OSException
```python 
OSException()
```



----


## Process
```python 
Process(
   info: typing.Union[Task, Service], loop: asyncio.AbstractEventLoop, **kwargs
)
```




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

----


## Module
```python 
Module(
   name
)
```



----


## FSModule
```python 
FSModule(
   root
)
```




**Methods:**


### .touch
```python
.touch(
   file
)
```


### .ls
```python
.ls(
   path
)
```


### .cd
```python
.cd(
   path
)
```


### .mkdir
```python
.mkdir()
```


### .rm
```python
.rm()
```


----


## OS
```python 
OS(
   root
)
```




**Methods:**


### .create_task
```python
.create_task(
   name, description, priority, status, mode, interval, script, arguments
)
```

