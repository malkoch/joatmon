#


## Task
```python 
Task(
   task, api, **kwargs
)
```


---
Task class for running an executable.

This class provides a way to run an executable with specified arguments.


**Attributes**

* **task** (str) : The task name.
* **api** (object) : The API object.
* **kwargs** (dict) : Additional keyword arguments.


**Args**

* **task** (str) : The task name.
* **api** (object) : The API object.
* **kwargs** (dict) : Additional keyword arguments.



**Methods:**


### .help
```python
.help()
```

---
Provide help about the 'run' function.


**Returns**

* **dict**  : A dictionary containing the name, description, and parameters of the 'run' function.


### .run
```python
.run()
```

---
Run the task.

This method runs the task by executing the specified executable with the provided arguments.


**Args**

* **executable** (str) : The executable to run.
* **args** (str) : The arguments to pass to the executable.

