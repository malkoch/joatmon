#


## Arithmetic
```python 
Arithmetic()
```


---
Enum representing arithmetic operations.


**Attributes**

* **add** (str) : Addition operation.
* **sub** (str) : Subtraction operation.
* **mul** (str) : Multiplication operation.
* **div** (str) : Division operation.


----


## Comparator
```python 
Comparator()
```


---
Base Enum for comparison operations.

----


## Equality
```python 
Equality()
```


---
Enum representing equality comparison operations.


**Attributes**

* **eq** (str) : Equality operation.
* **ne** (str) : Not equal operation.
* **gt** (str) : Greater than operation.
* **gte** (str) : Greater than or equal to operation.
* **lt** (str) : Less than operation.
* **lte** (str) : Less than or equal to operation.


----


## Matching
```python 
Matching()
```


---
Enum representing matching operations.


**Attributes**

* **not_like** (str) : Not like operation.
* **like** (str) : Like operation.


----


## Boolean
```python 
Boolean()
```


---
Enum representing boolean operations.


**Attributes**

* **and_** (str) : AND operation.
* **or_** (str) : OR operation.
* **xor_** (str) : XOR operation.
* **true** (str) : TRUE operation.
* **false** (str) : FALSE operation.


----


## Order
```python 
Order()
```


---
Enum representing order operations.


**Attributes**

* **asc** (str) : Ascending order.
* **desc** (str) : Descending order.


----


## JoinType
```python 
JoinType()
```


---
Enum representing join types in SQL.


**Attributes**

* **inner** (str) : Inner join.
* **left** (str) : Left join.
* **right** (str) : Right join.
* **outer** (str) : Full outer join.
* **left_outer** (str) : Left outer join.
* **right_outer** (str) : Right outer join.
* **full_outer** (str) : Full outer join.
* **cross** (str) : Cross join.


----


## DatePart
```python 
DatePart()
```


---
Enum representing date parts in SQL.


**Attributes**

* **year** (str) : Year part.
* **quarter** (str) : Quarter part.
* **month** (str) : Month part.
* **week** (str) : Week part.
* **day** (str) : Day part.
* **hour** (str) : Hour part.
* **minute** (str) : Minute part.
* **second** (str) : Second part.
* **microsecond** (str) : Microsecond part.


----


## Dialects
```python 
Dialects()
```


---
Enum representing SQL dialects.


**Attributes**

* **MSSQL** (str) : Microsoft SQL Server.
* **MYSQL** (str) : MySQL.
* **POSTGRESQL** (str) : PostgreSQL.
* **SQLLITE** (str) : SQLite.
* **MONGO** (str) : MongoDB.


----


## Node
```python 
Node(
   alias
)
```


---
Base class for a node in the query.


**Attributes**

* **alias** (str) : Alias for the node.



**Methods:**


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias for the node.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **Node**  : The node with the alias set.


----


## Term
```python 
Term(
   alias
)
```


---
Base class for a term in the query.


**Attributes**

* **alias** (str) : Alias for the term.



**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the term for the specified dialect.


**Args**

* **dialect** (Dialects) : The dialect to build the term for.
* **depth** (int) : The depth of the term in the query.


**Raises**

* **NotImplementedError**  : This method must be implemented by subclasses.


----


## ValueWrapper
```python 
ValueWrapper(
   value
)
```


---
Wrapper for a value in the query.


**Attributes**

* **value** (str, int, float, bool) : The value to wrap.



**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the value for the specified dialect.


**Args**

* **dialect** (Dialects) : The dialect to build the value for.
* **depth** (int) : The depth of the value in the query.


**Returns**

* **str**  : The built value.


----


## Criterion
```python 
Criterion()
```


---
Base class for a criterion in the query.


**Attributes**

* **alias** (str) : Alias for the criterion.



**Methods:**


### .any
```python
.any()
```


### .all
```python
.all()
```


----


## BasicCriteria
```python 
BasicCriteria(
   comparator: Comparator, left: Term, right: Term, alias = None
)
```


---
Class representing a basic criterion in the query.


**Attributes**

* **comparator** (Comparator) : The comparator for the criterion.
* **left** (Term) : The left term of the criterion.
* **right** (Term) : The right term of the criterion.
* **alias** (str) : Alias for the criterion.



**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the criterion for the specified dialect.


**Args**

* **dialect** (Dialects) : The dialect to build the criterion for.
* **depth** (int) : The depth of the criterion in the query.


**Raises**

* **ValueError**  : If the comparator is not supported by the dialect.


**Returns**

* **str**  : The built criterion.


----


## ComplexCriteria
```python 
ComplexCriteria(
   comparator: Comparator, left: Term, right: Term, alias = None
)
```


---
Class representing a complex criterion in the query.


**Attributes**

* **comparator** (Comparator) : The comparator for the criterion.
* **left** (Term) : The left term of the criterion.
* **right** (Term) : The right term of the criterion.
* **alias** (str) : Alias for the criterion.



**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the criterion for the specified dialect.


**Args**

* **dialect** (Dialects) : The dialect to build the criterion for.
* **depth** (int) : The depth of the criterion in the query.


**Returns**

* **str**  : The built criterion.


----


## ArithmeticExpression
```python 
ArithmeticExpression(
   operator: Arithmetic, left, right, alias = None
)
```


---
Class representing an arithmetic expression in the query.


**Attributes**

* **operator** (Arithmetic) : The operator for the expression.
* **left** (Term) : The left term of the expression.
* **right** (Term) : The right term of the expression.
* **alias** (str) : Alias for the expression.



**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the expression for the specified dialect.


**Args**

* **dialect** (Dialects) : The dialect to build the expression for.
* **depth** (int) : The depth of the expression in the query.


**Raises**

* **ValueError**  : If the operator is not supported by the dialect.


**Returns**

* **str**  : The built expression.


----


## Column
```python 
Column(
   name, table
)
```


---
Class representing a column in the query.


**Attributes**

* **_name** (str) : The name of the column.
* **_table** (Table) : The table the column belongs to.
* **alias** (str) : Alias for the column.



**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the column for the specified dialect.


**Args**

* **dialect** (Dialects) : The dialect to build the column for.
* **depth** (int) : The depth of the column in the query.


**Returns**

* **str**  : The built column.


----


## Table
```python 
Table(
   name, schema = None
)
```


---
Class representing a table in the database.


**Attributes**

* **_name** (str) : The name of the table.
* **_schema** (Schema) : The schema the table belongs to.
* **alias** (str) : The alias of the table.



**Methods:**


### .star
```python
.star()
```

---
Returns a Column object representing all columns in the table.


**Returns**

* **Column**  : A Column object representing all columns.


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias of the table.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **Table**  : The current table instance.


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the table.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the table.


----


## Schema
```python 
Schema(
   name, database
)
```


---
Class representing a schema in the database.


**Attributes**

* **_name** (str) : The name of the schema.
* **_database** (Database) : The database the schema belongs to.
* **alias** (str) : The alias of the schema.



**Methods:**


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias of the schema.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **Schema**  : The current schema instance.


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the schema.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the schema.


----


## Database
```python 
Database(
   name
)
```


---
Class representing a database.


**Attributes**

* **_name** (str) : The name of the database.
* **alias** (str) : The alias of the database.



**Methods:**


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias of the database.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **Database**  : The current database instance.


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the database.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the database.


----


## Count
```python 
Count(
   column
)
```


---
Class representing a count operation in SQL.


**Attributes**

* **column** (Column) : The column to count.
* **alias** (str) : The alias of the count operation.



**Methods:**


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias of the count operation.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **Count**  : The current count instance.


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the count operation.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the count operation.


----


## Array
```python 
Array(
   *column
)
```


---
Class representing an array in SQL.


**Attributes**

* **column** (list) : The columns to include in the array.
* **alias** (str) : The alias of the array.
* **dtype** (str) : The data type of the array.



**Methods:**


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias of the array.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **Array**  : The current array instance.


### .as_text
```python
.as_text()
```

---
Sets the data type of the array to text.


**Returns**

* **Array**  : The current array instance.


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the array.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the array.


----


## JSON
```python 
JSON(
   **kwargs
)
```


---
Class representing a JSON object in SQL.


**Attributes**

* **column** (dict) : The columns to include in the JSON object.
* **alias** (str) : The alias of the JSON object.
* **dtype** (str) : The data type of the JSON object.
* **is_array** (bool) : Whether the JSON object is an array.



**Methods:**


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias of the JSON object.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **JSON**  : The current JSON instance.


### .as_text
```python
.as_text()
```

---
Sets the data type of the JSON object to text.


**Returns**

* **JSON**  : The current JSON instance.


### .array
```python
.array()
```

---
Sets the JSON object to be an array.


**Returns**

* **JSON**  : The current JSON instance.


### .object
```python
.object()
```

---
Sets the JSON object to be an object.


**Returns**

* **JSON**  : The current JSON instance.


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the JSON object.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the JSON object.


----


## Sum
```python 
Sum(
   column
)
```


---
Class representing a sum operation in SQL.


**Attributes**

* **column** (Column) : The column to sum.



**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the sum operation.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the sum operation.


----


## Query
```python 

```


---
Class representing a SQL query.


**Attributes**

* **projection** (list) : The columns to include in the query.
* **condition** (Criterion) : The condition of the query.
* **grouping** (list) : The columns to group by in the query.
* **sort** (list) : The columns to sort by in the query.
* **limit** (int) : The limit of the query.
* **tables** (list) : The tables to include in the query.
* **joins** (list) : The joins to include in the query.
* **withs** (list) : The with clauses to include in the query.
* **alias** (str) : The alias of the query.



**Methods:**


### .as_table
```python
.as_table()
```

---
Returns a Table object representing the query.


**Returns**

* **Table**  : A Table object.


### .as_
```python
.as_(
   alias
)
```

---
Sets the alias of the query.


**Args**

* **alias** (str) : The alias to set.


**Returns**

* **Query**  : The current query instance.


### .with_
```python
.with_(
   *query
)
```

---
Adds with clauses to the query.


**Args**

* **query** (Query) : The queries to include in the with clause.


**Returns**

* **Query**  : The current query instance.


### .select
```python
.select(
   *terms
)
```

---
Adds columns to the projection of the query.


**Args**

* **terms** (Term) : The terms to include in the projection.


**Returns**

* **Query**  : The current query instance.


### .from_
```python
.from_(
   *tables
)
```

---
Adds tables to the query.


**Args**

* **tables** (Table) : The tables to include in the query.


**Returns**

* **Query**  : The current query instance.


### .join
```python
.join(
   table, term
)
```

---
Adds an inner join to the query.


**Args**

* **table** (Table) : The table to join.
* **term** (Term) : The condition of the join.


**Returns**

* **Query**  : The current query instance.


### .left_join
```python
.left_join(
   table, term
)
```

---
Adds a left join to the query.


**Args**

* **table** (Table) : The table to join.
* **term** (Term) : The condition of the join.


**Returns**

* **Query**  : The current query instance.


### .where
```python
.where(
   term
)
```

---
Adds a condition to the query.


**Args**

* **term** (Term) : The condition to add.


**Returns**

* **Query**  : The current query instance.


### .group
```python
.group(
   *terms
)
```

---
Adds terms to the grouping of the query.


**Args**

* **terms** (Term) : The terms to include in the grouping.


**Returns**

* **Query**  : The current query instance.


### .order
```python
.order(
   *terms
)
```

---
Adds terms to the sorting of the query.


**Args**

* **terms** (Term) : The terms to include in the sorting.


**Returns**

* **Query**  : The current query instance.


### .top
```python
.top(
   limit
)
```

---
Sets the limit of the query.


**Args**

* **limit** (int) : The limit to set.


**Returns**

* **Query**  : The current query instance.


### .build
```python
.build(
   dialect, depth = 0
)
```

---
Builds the SQL representation of the query.


**Args**

* **dialect** (Dialects) : The SQL dialect to use.
* **depth** (int, optional) : The depth of the query. Defaults to 0.


**Returns**

* **str**  : The SQL representation of the query.


**Raises**

* **ValueError**  : If the projection or tables of the query are empty.

