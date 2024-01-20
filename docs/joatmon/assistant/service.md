#


## Service
```python 
Service()
```


---
Service class for managing services.

This class provides a way to manage services, including their id, name, description, priority, status, mode, script, arguments, and timestamps.


**Attributes**

* **id** (str) : The unique identifier for the service.
* **name** (str) : The name of the service.
* **description** (str) : The description of the service.
* **priority** (int) : The priority of the service.
* **status** (str) : The status of the service.
* **mode** (str) : The mode of the service.
* **script** (str) : The script of the service.
* **arguments** (dict) : The arguments of the service.
* **created_at** (datetime.datetime) : The timestamp when the service was created.
* **updated_at** (datetime.datetime) : The timestamp when the service was last updated.


----


## ServiceState
```python 
ServiceState()
```


---
ServiceState class for managing the state of services.

This class provides a way to manage the state of services, including running, finished, and stopped states.


**Attributes**

* **running** (enum.auto) : The state when the service is running.
* **finished** (enum.auto) : The state when the service is finished.
* **stopped** (enum.auto) : The state when the service is stopped.


----


## ServiceInfo
```python 
ServiceInfo()
```


---
ServiceInfo class for managing the information of services.

This class provides a way to manage the information of services, including the service, its state, and its runnable.


**Attributes**

* **service** (Service) : The service.
* **state** (ServiceState) : The state of the service.
* **runnable** (Runnable) : The runnable of the service.


----


## BaseService
```python 
BaseService(
   service: Service, api, **kwargs
)
```


---
BaseService class for managing base services.

This class provides a way to manage base services, including their service and API.


**Attributes**

* **service** (Service) : The service.
* **api** (object) : The API object.


**Args**

* **service** (Service) : The service.
* **api** (object) : The API object.
* **kwargs** (dict) : Additional keyword arguments.

