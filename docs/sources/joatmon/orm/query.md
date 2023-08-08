#


## Arithmetic
```python 
Arithmetic()
```



----


## Comparator
```python 
Comparator()
```



----


## Equality
```python 
Equality()
```



----


## Matching
```python 
Matching()
```



----


## Boolean
```python 
Boolean()
```



----


## Order
```python 
Order()
```



----


## JoinType
```python 
JoinType()
```



----


## DatePart
```python 
DatePart()
```



----


## Dialects
```python 
Dialects()
```



----


## Node
```python 
Node(
   alias
)
```




**Methods:**


### .as_
```python
.as_(
   alias
)
```


----


## Term
```python 
Term(
   alias
)
```




**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## ValueWrapper
```python 
ValueWrapper(
   value
)
```




**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Criterion
```python 
Criterion()
```




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




**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## ComplexCriteria
```python 
ComplexCriteria(
   comparator: Comparator, left: Term, right: Term, alias = None
)
```




**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## ArithmeticExpression
```python 
ArithmeticExpression(
   operator: Arithmetic, left, right, alias = None
)
```




**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Column
```python 
Column(
   name, table
)
```




**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Table
```python 
Table(
   name, schema = None
)
```




**Methods:**


### .star
```python
.star()
```


### .as_
```python
.as_(
   alias
)
```


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Schema
```python 
Schema(
   name, database
)
```




**Methods:**


### .as_
```python
.as_(
   alias
)
```


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Database
```python 
Database(
   name
)
```




**Methods:**


### .as_
```python
.as_(
   alias
)
```


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Count
```python 
Count(
   column
)
```




**Methods:**


### .as_
```python
.as_(
   alias
)
```


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Array
```python 
Array(
   *column
)
```




**Methods:**


### .as_
```python
.as_(
   alias
)
```


### .as_text
```python
.as_text()
```


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## JSON
```python 
JSON(
   **kwargs
)
```




**Methods:**


### .as_
```python
.as_(
   alias
)
```


### .as_text
```python
.as_text()
```


### .array
```python
.array()
```


### .object
```python
.object()
```


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Sum
```python 
Sum(
   column
)
```




**Methods:**


### .build
```python
.build(
   dialect, depth = 0
)
```


----


## Query
```python 

```




**Methods:**


### .as_table
```python
.as_table()
```


### .as_
```python
.as_(
   alias
)
```


### .with_
```python
.with_(
   *query
)
```


### .select
```python
.select(
   *terms
)
```


### .from_
```python
.from_(
   *tables
)
```


### .join
```python
.join(
   table, term
)
```


### .left_join
```python
.left_join(
   table, term
)
```


### .where
```python
.where(
   term
)
```


### .group
```python
.group(
   *terms
)
```


### .order
```python
.order(
   *terms
)
```


### .top
```python
.top(
   limit
)
```


### .build
```python
.build(
   dialect, depth = 0
)
```

