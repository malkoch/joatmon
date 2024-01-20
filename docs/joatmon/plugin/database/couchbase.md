#


## CouchBaseDatabase
```python 
CouchBaseDatabase(
   uri, bucket, scope, username, password
)
```


---
CouchBaseDatabase class that inherits from the DatabasePlugin class. It implements the abstract methods of the DatabasePlugin class
using Couchbase for database operations.


**Attributes**

* **DATABASES** (set) : A set to store the databases.
* **CREATED_COLLECTIONS** (set) : A set to store the created collections.
* **UPDATED_COLLECTIONS** (set) : A set to store the updated collections.
* **bucket** (str) : The bucket of the Couchbase server.
* **scope** (str) : The scope of the Couchbase server.
* **db** (`couchbase.cluster.Cluster` instance) : The connection to the Couchbase server.



**Methods:**


### .insert
```python
.insert(
   document, *docs
)
```

---
Insert one or more documents into the Couchbase server.


**Args**

* **document** (dict) : The first document to be inserted.
* **docs** (dict) : Additional documents to be inserted.


### .read
```python
.read(
   document, query
)
```

---
Read a document from the Couchbase server.


**Args**

* **document** (dict) : The document to be read.
* **query** (dict) : The query to be used for reading the document.


**Yields**

* **dict**  : The read document.


### .update
```python
.update(
   document, query, update
)
```

---
Update a document in the Couchbase server.


**Args**

* **document** (dict) : The document to be updated.
* **query** (dict) : The query to be used for updating the document.
* **update** (dict) : The update to be applied to the document.


### .delete
```python
.delete(
   document, query
)
```

---
Delete a document from the Couchbase server.


**Args**

* **document** (dict) : The document to be deleted.
* **query** (dict) : The query to be used for deleting the document.


### .start
```python
.start()
```

---
Start a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .commit
```python
.commit()
```

---
Commit a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .abort
```python
.abort()
```

---
Abort a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.


### .end
```python
.end()
```

---
End a database transaction.


**Raises**

* **NotImplementedError**  : This method should be implemented in the child classes.

