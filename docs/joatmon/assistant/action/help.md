#


## Task
```python 
Task(
   task, api, **kwargs
)
```


---
Task class for providing help about a function.

This class provides a way to learn about a function by returning its name, description, and parameters.


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
Provide help about the 'help' function.


**Returns**

* **dict**  : A dictionary containing the name, description, and parameters of the 'help' function.


### .run
```python
.run()
```

---
Run the task.

This method runs the task by loading the scripts from the API folders, importing the modules, and getting the help about the functions in the modules.
