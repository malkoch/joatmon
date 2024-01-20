#


## Localizer
```python 
Localizer()
```


---
Localizer class that inherits from the Plugin class. It is an abstract class that provides
the structure for localization operations. The methods in this class should be implemented in the child classes.


**Methods:**


### .localize
```python
.localize(
   language, value
)
```

---
This method is used to localize a given value to a specified language.


**Args**

* **language** (str) : The language to which the value should be localized.
* **value** (str) : The value to be localized.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.

