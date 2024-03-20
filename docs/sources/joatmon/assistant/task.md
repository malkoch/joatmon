#


## Task
```python 
Task()
```


---
Task class for managing tasks.

This class provides a way to manage tasks, including their id, name, description, priority, status, script, arguments, and timestamps.


**Attributes**

* **id** (str) : The unique identifier for the task.
* **name** (str) : The name of the task.
* **description** (str) : The description of the task.
* **priority** (int) : The priority of the task.
* **status** (bool) : The status of the task.
* **on** (str) : The script of the task.
* **arguments** (dict) : The arguments of the task.
* **created_at** (datetime.datetime) : The timestamp when the task was created.
* **updated_at** (datetime.datetime) : The timestamp when the task was last updated.


----


## TaskState
```python 
TaskState()
```


---
TaskState class for managing the state of tasks.

This class provides a way to manage the state of tasks, including running and finished states.


**Attributes**

* **running** (enum.auto) : The state when the task is running.
* **finished** (enum.auto) : The state when the task is finished.


----


## TaskInfo
```python 
TaskInfo()
```


---
TaskInfo class for managing the information of tasks.

This class provides a way to manage the information of tasks, including the task, its state, and its runnable.


**Attributes**

* **task** (Task) : The task.
* **state** (TaskState) : The state of the task.
* **runnable** (Runnable) : The runnable of the task.


----


## BaseTask
```python 
BaseTask(
   task: Task, api, **kwargs
)
```


---
BaseTask class for managing base tasks.

This class provides a way to manage base tasks, including their task and API.


**Attributes**

* **info** (Task) : The task.
* **api** (API) : The API object.


**Args**

* **task** (Task) : The task.
* **api** (API) : The API object.
* **kwargs** (dict) : Additional keyword arguments.

