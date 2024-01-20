#


## PostgreSQLDatabase
```python 
PostgreSQLDatabase(
   host, port, user, password, database
)
```


---
PostgreSQLDatabase class that inherits from the DatabasePlugin class. It implements the abstract methods of the DatabasePlugin class
using PostgreSQL for database operations.


**Attributes**

* **DATABASES** (set) : A set to store the databases.
* **CREATED_COLLECTIONS** (set) : A set to store the created collections.
* **UPDATED_COLLECTIONS** (set) : A set to store the updated collections.
* **connection** (`psycopg2.extensions.connection` instance) : The connection to the PostgreSQL server.



**Methods:**


### .create
```python
.create(
   document
)
```

---
Create a new document in the PostgreSQL database.


**Args**

* **document** (dict) : The document to be created.


### .alter
```python
.alter(
   document
)
```

---
Alter an existing document in the PostgreSQL database.


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
Drop a collection from the PostgreSQL database.


**Args**

* **document** (dict) : The document whose collection is to be dropped.


### .insert
```python
.insert(
   document, *docs
)
```

---
Insert one or more documents into the PostgreSQL database.


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
Read a document from the PostgreSQL database.


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
Update a document in the PostgreSQL database.


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
Delete a document from the PostgreSQL database.


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
View a document in the PostgreSQL database.


**Args**

* **document** (dict) : The document to be viewed.
* **query** (dict) : The query to be used for viewing the document.


**Yields**

* **dict**  : The viewed document.


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
Start a database transaction in the PostgreSQL database.

This method sets the autocommit mode of the connection to False, which means that changes made to the database
are not saved until you call the commit method.

### .commit
```python
.commit()
```

---
Commit a database transaction in the PostgreSQL database.

This method saves the changes made to the database since the last call to the start method.

### .abort
```python
.abort()
```

---
Abort a database transaction in the PostgreSQL database.

This method discards the changes made to the database since the last call to the start method.

### .end
```python
.end()
```

---
End a database transaction in the PostgreSQL database.

This method closes the connection to the database. After calling this method, you cannot make any more
queries to the database using this connection.
