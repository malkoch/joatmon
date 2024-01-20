#


## DatabasePlugin
```python 
DatabasePlugin()
```


---
DatabasePlugin class that inherits from the Plugin class. It is an abstract class that provides
the structure for database operations. The methods in this class should be implemented in the child classes.


**Methods:**


### .create
```python
.create(
   document
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
create a new document in the database.


**Args**

* **document** (dict) : The document to be created.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .alter
```python
.alter(
   document
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
alter an existing document in the database.


**Args**

* **document** (dict) : The document to be altered.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .drop
```python
.drop(
   document
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
drop a document from the database.


**Args**

* **document** (dict) : The document to be dropped.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .insert
```python
.insert(
   document, *docs
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
insert one or more documents into the database.


**Args**

* **document** (dict) : The first document to be inserted.
* **docs** (dict) : Additional documents to be inserted.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .read
```python
.read(
   document, query
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
read a document from the database.


**Args**

* **document** (dict) : The document to be read.
* **query** (dict) : The query to be used for reading the document.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .update
```python
.update(
   document, query, update
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
update a document in the database.


**Args**

* **document** (dict) : The document to be updated.
* **query** (dict) : The query to be used for updating the document.
* **update** (dict) : The update to be applied to the document.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .delete
```python
.delete(
   document, query
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
delete a document from the database.


**Args**

* **document** (dict) : The document to be deleted.
* **query** (dict) : The query to be used for deleting the document.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .view
```python
.view(
   document, query
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
view a document in the database.


**Args**

* **document** (dict) : The document to be viewed.
* **query** (dict) : The query to be used for viewing the document.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .execute
```python
.execute(
   document, query
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
execute a query on a document in the database.


**Args**

* **document** (dict) : The document to be queried.
* **query** (dict) : The query to be executed.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .count
```python
.count(
   query
)
```

---
This is an abstract method that should be implemented in the child classes. It is used to
count the number of documents that match a query in the database.


**Args**

* **query** (dict) : The query to be counted.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .start
```python
.start()
```

---
This is an abstract method that should be implemented in the child classes. It is used to
start a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .commit
```python
.commit()
```

---
This is an abstract method that should be implemented in the child classes. It is used to
commit a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .abort
```python
.abort()
```

---
This is an abstract method that should be implemented in the child classes. It is used to
abort a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .end
```python
.end()
```

---
This is an abstract method that should be implemented in the child classes. It is used to
end a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.

