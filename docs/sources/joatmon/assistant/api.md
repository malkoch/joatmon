#


## API
```python 
API(
   loop, cwd, folders = None, tasks: typing.Optional[typing.List[Task]] = None,
   services: typing.Optional[typing.List[Service]] = None,
   events: typing.Optional[typing.Dict[str, Event]] = None
)
```


---
API class for managing tasks and services.

This class provides a way to manage tasks and services, including running tasks, starting and stopping services, and cleaning up processes.


**Attributes**

* **loop** (asyncio.AbstractEventLoop) : The event loop where the tasks and services are run.
* **cwd** (str) : The current working directory.
* **folders** (list) : The folders where the scripts for tasks and services are located.
* **tasks** (list) : The list of tasks.
* **services** (list) : The list of services.
* **processes** (list) : The list of running processes.
* **event** (asyncio.Event) : An event for signaling the termination of the API.


**Args**

* **loop** (asyncio.AbstractEventLoop) : The event loop where the tasks and services are run.
* **cwd** (str) : The current working directory.
* **folders** (list, optional) : The folders where the scripts for tasks and services are located.
* **tasks** (list, optional) : The list of tasks.
* **services** (list, optional) : The list of services.



**Methods:**


### .main
```python
.main()
```

---
Main method for running the API.

This method starts the tasks and services, and waits for them to complete. It also handles cleaning up the processes.

### .run_interval
```python
.run_interval()
```

---
Run tasks at specified intervals.

This method runs the tasks that are configured to run at specified intervals.

### .run_services
```python
.run_services()
```

---
Run services.

This method runs the services that are configured to run automatically.

### .action
```python
.action(
   action, arguments
)
```

---
Perform an action.

This method performs an action based on the provided action name and arguments.


**Args**

* **action** (str) : The name of the action to perform.
* **arguments** (dict) : The arguments for the action.


### .run_task
```python
.run_task(
   task_name, kwargs = None
)
```

---
Run a task by name.

This method runs a task with the provided name and arguments.


**Args**

* **task_name** (str) : The name of the task to run.
* **kwargs** (dict, optional) : The arguments for the task.


### .start_service
```python
.start_service(
   service_name
)
```

---
Start a service by name.

This method starts a service with the provided name.


**Args**

* **service_name** (str) : The name of the service to start.


### .stop_service
```python
.stop_service(
   service_name
)
```

---
Stop a service by name.

This method stops a service with the provided name.


**Args**

* **service_name** (str) : The name of the service to stop.


### .restart_service
```python
.restart_service(
   service_name
)
```

---
Restart a service by name.

This method restarts a service with the provided name.


**Args**

* **service_name** (str) : The name of the service to restart.


### .exit
```python
.exit()
```

---
Exit the API.

This method stops all tasks and services, and signals the termination of the API.
