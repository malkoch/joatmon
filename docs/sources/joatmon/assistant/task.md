#


## BaseTask
```python 
BaseTask(
   api, background = False, **kwargs
)
```




**Methods:**


### .background
```python
.background()
```


### .help
```python
.help()
```


### .run
```python
.run()
```


### .running
```python
.running()
```


### .start
```python
.start()
```


### .stop
```python
.stop()
```


----


## TaskState
```python 
TaskState()
```



----


## TaskInfo
```python 
TaskInfo()
```



----


### on_begin
```python
.on_begin(
   name, *args, **kwargs
)
```


----


### on_error
```python
.on_error(
   name, *args, **kwargs
)
```


----


### on_end
```python
.on_end(
   name, *args, **kwargs
)
```


----


### create
```python
.create(
   api
)
```


----


### get_class
```python
.get_class(
   name
)
```


----


### get
```python
.get(
   api, name, kwargs, background
)
```

