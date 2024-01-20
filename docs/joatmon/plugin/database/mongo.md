#


## MongoDatabase
```python 
MongoDatabase(
   uri, database
)
```


---
MongoDatabase class that inherits from the DatabasePlugin class. It implements the abstract methods of the DatabasePlugin class
using MongoDB for database operations.


**Attributes**

* **DATABASES** (set) : A set to store the databases.
* **CREATED_COLLECTIONS** (set) : A set to store the created collections.
* **UPDATED_COLLECTIONS** (set) : A set to store the updated collections.
* **database_name** (str) : The name of the MongoDB database.
* **client** (`pymongo.MongoClient` instance) : The connection to the MongoDB server.
* **database** (`pymongo.database.Database` instance) : The MongoDB database instance.
* **session** (`pymongo.client_session.ClientSession` instance) : The MongoDB client session instance.



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
Drop a collection from the MongoDB database.


**Args**

* **document** (dict) : The document whose collection is to be dropped.


### .insert
```python
.insert(
   document, *docs
)
```

---
Insert one or more documents into the MongoDB database.


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
Read a document from the MongoDB database.


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
Update a document in the MongoDB database.


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
Delete a document from the MongoDB database.


**Args**

* **document** (dict) : The document to be deleted.
* **query** (dict) : The query to be used for deleting the document.


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
Start a database transaction in the MongoDB database.

### .commit
```python
.commit()
```

---
Commit a database transaction in the MongoDB database.

### .abort
```python
.abort()
```

---
Abort a database transaction in the MongoDB database.

### .end
```python
.end()
```

---
End a database transaction in the MongoDB database.
