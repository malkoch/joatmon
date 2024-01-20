#


## Plugin
```python 
Plugin()
```


---
Abstract Plugin class that defines the interface for plugins.

This class should be inherited by any class that wants to be used as a plugin.

----


## PluginProxy
```python 
PluginProxy(
   cls, *args, **kwargs
)
```


---
PluginProxy class that acts as a proxy for the actual plugin class.

This class is used to delay the instantiation of the actual plugin class until it is needed.


**Attributes**

* **cls** (class) : The class of the actual plugin.
* **args** (tuple) : The arguments for the plugin class.
* **kwargs** (dict) : The keyword arguments for the plugin class.
* **instance** (Plugin or None) : The instance of the actual plugin class.


----


### register
```python
.register(
   cls, alias, *args, **kwargs
)
```

---
Registers a plugin with a given alias.


**Args**

* **cls** (str or class) : The class or the fully qualified name of the class to be registered.
* **alias** (str) : The alias for the plugin.
* **args**  : Variable length argument list for the plugin class.
* **kwargs**  : Arbitrary keyword arguments for the plugin class.

