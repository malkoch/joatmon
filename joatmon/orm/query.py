from enum import Enum


class Arithmetic(Enum):
    """
    Enum representing arithmetic operations.

    Attributes:
        add (str): Addition operation.
        sub (str): Subtraction operation.
        mul (str): Multiplication operation.
        div (str): Division operation.
    """

    add = '+'
    sub = '-'
    mul = '*'
    div = '/'


class Comparator(Enum):
    """
    Base Enum for comparison operations.
    """


class Equality(Comparator):
    """
    Enum representing equality comparison operations.

    Attributes:
        eq (str): Equality operation.
        ne (str): Not equal operation.
        gt (str): Greater than operation.
        gte (str): Greater than or equal to operation.
        lt (str): Less than operation.
        lte (str): Less than or equal to operation.
    """

    eq = '='
    ne = '<>'
    gt = '>'
    gte = '>='
    lt = '<'
    lte = '<='


class Matching(Comparator):
    """
    Enum representing matching operations.

    Attributes:
        not_like (str): Not like operation.
        like (str): Like operation.
    """

    not_like = ' NOT LIKE '
    like = ' LIKE '


class Boolean(Comparator):
    """
    Enum representing boolean operations.

    Attributes:
        and_ (str): AND operation.
        or_ (str): OR operation.
        xor_ (str): XOR operation.
        true (str): TRUE operation.
        false (str): FALSE operation.
    """

    and_ = 'AND'
    or_ = 'OR'
    xor_ = 'XOR'
    true = 'TRUE'
    false = 'FALSE'


class Order(Enum):
    """
    Enum representing order operations.

    Attributes:
        asc (str): Ascending order.
        desc (str): Descending order.
    """

    asc = 'ASC'
    desc = 'DESC'


class JoinType(Enum):
    """
    Enum representing join types in SQL.

    Attributes:
        inner (str): Inner join.
        left (str): Left join.
        right (str): Right join.
        outer (str): Full outer join.
        left_outer (str): Left outer join.
        right_outer (str): Right outer join.
        full_outer (str): Full outer join.
        cross (str): Cross join.
    """

    inner = ''
    left = 'LEFT'
    right = 'RIGHT'
    outer = 'FULL OUTER'
    left_outer = 'LEFT OUTER'
    right_outer = 'RIGHT OUTER'
    full_outer = 'FULL OUTER'
    cross = 'CROSS'


class DatePart(Enum):
    """
    Enum representing date parts in SQL.

    Attributes:
        year (str): Year part.
        quarter (str): Quarter part.
        month (str): Month part.
        week (str): Week part.
        day (str): Day part.
        hour (str): Hour part.
        minute (str): Minute part.
        second (str): Second part.
        microsecond (str): Microsecond part.
    """

    year = 'YEAR'
    quarter = 'QUARTER'
    month = 'MONTH'
    week = 'WEEK'
    day = 'DAY'
    hour = 'HOUR'
    minute = 'MINUTE'
    second = 'SECOND'
    microsecond = 'MICROSECOND'


class Dialects(Enum):
    """
    Enum representing SQL dialects.

    Attributes:
        MSSQL (str): Microsoft SQL Server.
        MYSQL (str): MySQL.
        POSTGRESQL (str): PostgreSQL.
        SQLLITE (str): SQLite.
        MONGO (str): MongoDB.
    """

    MSSQL = 'mssql'
    MYSQL = 'mysql'
    POSTGRESQL = 'postgressql'
    SQLLITE = 'sqllite'
    MONGO = 'mongo'


class Node:
    """
    Base class for a node in the query.

    Attributes:
        alias (str): Alias for the node.
    """

    def __init__(self, alias):
        self.alias = alias

    def as_(self, alias):
        """
        Sets the alias for the node.

        Args:
            alias (str): The alias to set.

        Returns:
            Node: The node with the alias set.
        """
        self.alias = alias
        return self


class Term(Node):
    """
    Base class for a term in the query.

    Attributes:
        alias (str): Alias for the term.
    """

    def __init__(self, alias):
        super(Term, self).__init__(alias)

    def __eq__(self, other):
        return BasicCriteria(Equality.eq, self, other)

    def __ne__(self, other):
        return BasicCriteria(Equality.ne, self, other)

    def __gt__(self, other):
        return BasicCriteria(Equality.gt, self, other)

    def __ge__(self, other):
        return BasicCriteria(Equality.gte, self, other)

    def __lt__(self, other):
        return BasicCriteria(Equality.lt, self, other)

    def __le__(self, other):
        return BasicCriteria(Equality.lte, self, other)

    def __sub__(self, other):
        return ArithmeticExpression(Arithmetic.sub, self, other)

    def __add__(self, other):
        return ArithmeticExpression(Arithmetic.add, self, other)

    def __mul__(self, other):
        return ArithmeticExpression(Arithmetic.mul, self, other)

    def __truediv__(self, other):
        return ArithmeticExpression(Arithmetic.div, self, other)

    def build(self, dialect, depth=0):
        """
        Builds the term for the specified dialect.

        Args:
            dialect (Dialects): The dialect to build the term for.
            depth (int): The depth of the term in the query.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError


class ValueWrapper:
    """
    Wrapper for a value in the query.

    Attributes:
        value (str, int, float, bool): The value to wrap.
    """

    def __init__(self, value):
        self.value = value

    def build(self, dialect, depth=0):
        """
        Builds the value for the specified dialect.

        Args:
            dialect (Dialects): The dialect to build the value for.
            depth (int): The depth of the value in the query.

        Returns:
            str: The built value.
        """
        if self.value is None:
            return 'null'
        elif self.value is True:
            return 'true'
        elif self.value is False:
            return 'false'
        elif isinstance(self.value, str):
            return f"'{self.value}'"
        return str(self.value)


class Criterion(Term):
    """
    Base class for a criterion in the query.

    Attributes:
        alias (str): Alias for the criterion.
    """

    def __and__(self, other):
        return ComplexCriteria(Boolean.and_, self, other)

    def __or__(self, other):
        return ComplexCriteria(Boolean.or_, self, other)

    def __xor__(self, other):
        ...

    @staticmethod
    def any():
        ...

    @staticmethod
    def all():
        ...


class BasicCriteria(Criterion):
    """
    Class representing a basic criterion in the query.

    Attributes:
        comparator (Comparator): The comparator for the criterion.
        left (Term): The left term of the criterion.
        right (Term): The right term of the criterion.
        alias (str): Alias for the criterion.
    """

    def __init__(self, comparator: Comparator, left: Term, right: Term, alias=None):
        super(BasicCriteria, self).__init__(alias)

        self.comparator = comparator
        self.left = left
        if right is None:
            right = ValueWrapper(right)
        if isinstance(right, (bool, int, float, str)):
            right = ValueWrapper(right)
        self.right = right

    def build(self, dialect, depth=0):
        """
        Builds the criterion for the specified dialect.

        Args:
            dialect (Dialects): The dialect to build the criterion for.
            depth (int): The depth of the criterion in the query.

        Raises:
            ValueError: If the comparator is not supported by the dialect.

        Returns:
            str: The built criterion.
        """
        if dialect == Dialects.MSSQL:
            ret = ''
            if self.comparator == Equality.eq:
                return f'{self.left.build(dialect)} = {self.right.build(dialect)}'
            if self.comparator == Equality.ne:
                return f'{self.left.build(dialect)} != {self.right.build(dialect)}'
            if self.comparator == Equality.gt:
                return f'{self.left.build(dialect)} > {self.right.build(dialect)}'
            if self.comparator == Equality.gte:
                return f'{self.left.build(dialect)} >= {self.right.build(dialect)}'
            if self.comparator == Equality.lt:
                return f'{self.left.build(dialect)} < {self.right.build(dialect)}'
            if self.comparator == Equality.lte:
                return f'{self.left.build(dialect)} <= {self.right.build(dialect)}'
            return ret
        if dialect == Dialects.POSTGRESQL:
            ret = ''
            if self.comparator == Equality.eq:
                if isinstance(self.right, ValueWrapper) and self.right.value is None:
                    ret += f'{self.left.build(dialect)} is {self.right.build(dialect)}'
                else:
                    ret += f'{self.left.build(dialect)} = {self.right.build(dialect)}'
            if self.comparator == Equality.ne:
                if isinstance(self.right, ValueWrapper) and self.right.value is None:
                    ret += f'{self.left.build(dialect)} is not {self.right.build(dialect)}'
                else:
                    ret += f'{self.left.build(dialect)} != {self.right.build(dialect)}'
            if self.comparator == Equality.gt:
                ret += f'{self.left.build(dialect)} > {self.right.build(dialect)}'
            if self.comparator == Equality.gte:
                ret += f'{self.left.build(dialect)} >= {self.right.build(dialect)}'
            if self.comparator == Equality.lt:
                ret += f'{self.left.build(dialect)} < {self.right.build(dialect)}'
            if self.comparator == Equality.lte:
                ret += f'{self.left.build(dialect)} <= {self.right.build(dialect)}'
            return ret if self.alias is None else f'{ret} as {self.alias}'
        if dialect == Dialects.MONGO:
            ret = {}
            if self.comparator == Equality.eq:
                ret[self.left.build(dialect)] = {'$eq': self.right.build(dialect)}
            print(ret)
            return ret


class ComplexCriteria(Criterion):
    """
    Class representing a complex criterion in the query.

    Attributes:
        comparator (Comparator): The comparator for the criterion.
        left (Term): The left term of the criterion.
        right (Term): The right term of the criterion.
        alias (str): Alias for the criterion.
    """

    def __init__(self, comparator: Comparator, left: Term, right: Term, alias=None):
        super(ComplexCriteria, self).__init__(alias)

        self.comparator = comparator
        self.left = left
        self.right = right

    def build(self, dialect, depth=0):
        """
        Builds the criterion for the specified dialect.

        Args:
            dialect (Dialects): The dialect to build the criterion for.
            depth (int): The depth of the criterion in the query.

        Returns:
            str: The built criterion.
        """
        if dialect == Dialects.MSSQL:
            ret = ''
            if self.comparator == Boolean.and_:
                return f'{self.left.build(dialect)} and \n{self.right.build(dialect)}'
            if self.comparator == Boolean.or_:
                return f'{self.left.build(dialect)} or \n{self.right.build(dialect)}'
            return ret
        if dialect == Dialects.POSTGRESQL:
            ret = ''
            if self.comparator == Boolean.and_:
                ret += f'{self.left.build(dialect)} and \n{self.right.build(dialect)}'
            if self.comparator == Boolean.or_:
                ret += f'{self.left.build(dialect)} or \n{self.right.build(dialect)}'
            return ret if self.alias is None else f'{ret} as {self.alias}'
        if dialect == Dialects.MONGO:
            ret = {}
            if self.comparator == Boolean.and_:
                ret['$and'] = [self.left.build(dialect), self.right.build(dialect)]
            print(ret)
            return ret


class ArithmeticExpression(Term):
    """
    Class representing an arithmetic expression in the query.

    Attributes:
        operator (Arithmetic): The operator for the expression.
        left (Term): The left term of the expression.
        right (Term): The right term of the expression.
        alias (str): Alias for the expression.
    """

    def __init__(self, operator: Arithmetic, left, right, alias=None):
        super(ArithmeticExpression, self).__init__(alias)

        self.operator = operator
        self.left = left
        if isinstance(right, (bool, int, float, str)):
            right = ValueWrapper(right)
        self.right = right

    def build(self, dialect, depth=0):
        """
        Builds the expression for the specified dialect.

        Args:
            dialect (Dialects): The dialect to build the expression for.
            depth (int): The depth of the expression in the query.

        Raises:
            ValueError: If the operator is not supported by the dialect.

        Returns:
            str: The built expression.
        """
        if dialect == Dialects.MSSQL:
            ret = '('
            if self.operator == Arithmetic.add:
                ret += f'{self.left.build(dialect)} + {self.right.build(dialect)}'
            if self.operator == Arithmetic.sub:
                ret += f'{self.left.build(dialect)} - {self.right.build(dialect)}'
            if self.operator == Arithmetic.mul:
                ret += f'{self.left.build(dialect)} * {self.right.build(dialect)}'
            if self.operator == Arithmetic.div:
                ret += f'{self.left.build(dialect)} / {self.right.build(dialect)}'
            ret += ')'
            return ret if self.alias is None else f'{ret} as {self.alias}'
        if dialect == Dialects.POSTGRESQL:
            ret = '('
            if self.operator == Arithmetic.add:
                ret += f'{self.left.build(dialect)} + {self.right.build(dialect)}'
            if self.operator == Arithmetic.sub:
                ret += f'{self.left.build(dialect)} - {self.right.build(dialect)}'
            if self.operator == Arithmetic.mul:
                ret += f'{self.left.build(dialect)} * {self.right.build(dialect)}'
            if self.operator == Arithmetic.div:
                ret += f'{self.left.build(dialect)} / {self.right.build(dialect)}'
            ret += ')'
            return ret if self.alias is None else f'{ret} as {self.alias}'
        if dialect == Dialects.MONGO:
            ret = {}
            if self.operator == Arithmetic.sub:
                ret['$add'] = [self.left.build(dialect), self.right.build(dialect)]
            print(ret)
            return ret


class Column(Term):
    """
    Class representing a column in the query.

    Attributes:
        _name (str): The name of the column.
        _table (Table): The table the column belongs to.
        alias (str): Alias for the column.
    """

    def __init__(self, name, table):
        super(Column, self).__init__(None)

        self._name = name
        self._table = table
        self.alias = None

    def build(self, dialect, depth=0):
        """
        Builds the column for the specified dialect.

        Args:
            dialect (Dialects): The dialect to build the column for.
            depth (int): The depth of the column in the query.

        Returns:
            str: The built column.
        """
        if dialect == Dialects.POSTGRESQL:
            return f'{self._table.build(dialect)}."{self._name}"'


class Table:
    """
    Class representing a table in the database.

    Attributes:
        _name (str): The name of the table.
        _schema (Schema): The schema the table belongs to.
        alias (str): The alias of the table.
    """

    def __init__(self, name, schema=None):
        """
        Initializes a new instance of the Table class.

        Args:
            name (str): The name of the table.
            schema (Schema, optional): The schema the table belongs to. Defaults to None.
        """
        self._name = name
        self._schema = schema
        self.alias = None

    def __getattr__(self, item):
        """
        Returns a Column object with the given item as name and the current table as parent.

        Args:
            item (str): The name of the column.

        Returns:
            Column: A Column object.
        """
        return Column(item, self)

    def star(self):
        """
        Returns a Column object representing all columns in the table.

        Returns:
            Column: A Column object representing all columns.
        """
        return Column('*', self)

    def as_(self, alias):
        """
        Sets the alias of the table.

        Args:
            alias (str): The alias to set.

        Returns:
            Table: The current table instance.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the table.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the table.
        """
        if dialect == Dialects.POSTGRESQL:
            if self._schema is None:
                return f'"{self._name}"'
            else:
                return f'{self._schema.build(dialect)}."{self._name}"'


class Schema:
    """
    Class representing a schema in the database.

    Attributes:
        _name (str): The name of the schema.
        _database (Database): The database the schema belongs to.
        alias (str): The alias of the schema.
    """

    def __init__(self, name, database):
        """
        Initializes a new instance of the Schema class.

        Args:
            name (str): The name of the schema.
            database (Database): The database the schema belongs to.
        """
        self._name = name
        self._database = database
        self.alias = None

    def __getattr__(self, item):
        """
        Returns a Table object with the given item as name and the current schema as parent.

        Args:
            item (str): The name of the table.

        Returns:
            Table: A Table object.
        """
        return Table(item, self)

    def as_(self, alias):
        """
        Sets the alias of the schema.

        Args:
            alias (str): The alias to set.

        Returns:
            Schema: The current schema instance.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the schema.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the schema.
        """
        if dialect == Dialects.POSTGRESQL:
            return f'{self._database.build(dialect)}."{self._name}"'


class Database:
    """
    Class representing a database.

    Attributes:
        _name (str): The name of the database.
        alias (str): The alias of the database.
    """

    def __init__(self, name):
        """
        Initializes a new instance of the Database class.

        Args:
            name (str): The name of the database.
        """
        self._name = name
        self.alias = None

    def __getattr__(self, item):
        """
        Returns a Schema object with the given item as name and the current database as parent.

        Args:
            item (str): The name of the schema.

        Returns:
            Schema: A Schema object.
        """
        return Schema(item, self)

    def as_(self, alias):
        """
        Sets the alias of the database.

        Args:
            alias (str): The alias to set.

        Returns:
            Database: The current database instance.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the database.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the database.
        """
        if dialect == Dialects.POSTGRESQL:
            return f'"{self._name}"'


class Count:
    """
    Class representing a count operation in SQL.

    Attributes:
        column (Column): The column to count.
        alias (str): The alias of the count operation.
    """

    def __init__(self, column):
        """
        Initializes a new instance of the Count class.

        Args:
            column (Column): The column to count.
        """
        self.column = column
        self.alias = None

    def as_(self, alias):
        """
        Sets the alias of the count operation.

        Args:
            alias (str): The alias to set.

        Returns:
            Count: The current count instance.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the count operation.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the count operation.
        """
        if dialect == Dialects.MSSQL:
            if isinstance(self.column, Column):
                return f'count({self.column._table._name}.{self.column._name})'
            elif isinstance(self.column, Criterion):
                return f'count({self.column.build(dialect)})'
        if dialect == Dialects.POSTGRESQL:
            if isinstance(self.column, Column):
                return f'count({self.column._table._name}.{self.column._name})'
            elif isinstance(self.column, Criterion):
                return f'count(*) filter(where {self.column.build(dialect)})'
        if dialect == Dialects.MONGO:
            ...


class Array:
    """
    Class representing an array in SQL.

    Attributes:
        column (list): The columns to include in the array.
        alias (str): The alias of the array.
        dtype (str): The data type of the array.
    """

    def __init__(self, *column):
        """
        Initializes a new instance of the Array class.

        Args:
            *column (Column): The columns to include in the array.
        """
        self.column = column
        self.alias = None

        self.dtype = None

    def as_(self, alias):
        """
        Sets the alias of the array.

        Args:
            alias (str): The alias to set.

        Returns:
            Array: The current array instance.
        """
        self.alias = alias
        return self

    def as_text(self):
        """
        Sets the data type of the array to text.

        Returns:
            Array: The current array instance.
        """
        self.dtype = 'text'
        return self

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the array.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the array.
        """
        if dialect == Dialects.MSSQL:
            raise ValueError
        if dialect == Dialects.POSTGRESQL:
            if isinstance(self.column[0], Column):
                return f'array_agg(({",".join([x._table._name + "." + x._name for x in self.column])}))'
            elif isinstance(self.column[0], Table):
                return f'array_agg({self.column[0]._name})'
            elif isinstance(self.column[0], Query):
                # print('-' * 30)
                # print(self.column[0].build(dialect))
                # print('-' * 30)
                return f'ARRAY({self.column[0].build(dialect)}){f"::{self.dtype}[]" if self.dtype is not None else ""}'
            elif isinstance(self.column[0], Criterion):
                return f'count(*) filter(where {self.column[0].build(dialect)})'
        if dialect == Dialects.MONGO:
            ...


class JSON:
    """
    Class representing a JSON object in SQL.

    Attributes:
        column (dict): The columns to include in the JSON object.
        alias (str): The alias of the JSON object.
        dtype (str): The data type of the JSON object.
        is_array (bool): Whether the JSON object is an array.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the JSON class.

        Args:
            **kwargs (Column): The columns to include in the JSON object.
        """
        self.column = kwargs
        self.alias = None

        self.dtype = None
        self.is_array = True

    def as_(self, alias):
        """
        Sets the alias of the JSON object.

        Args:
            alias (str): The alias to set.

        Returns:
            JSON: The current JSON instance.
        """
        self.alias = alias
        return self

    def as_text(self):
        """
        Sets the data type of the JSON object to text.

        Returns:
            JSON: The current JSON instance.
        """
        self.dtype = 'text'
        return self

    def array(self):
        """
        Sets the JSON object to be an array.

        Returns:
            JSON: The current JSON instance.
        """
        self.is_array = True
        return self

    def object(self):
        """
        Sets the JSON object to be an object.

        Returns:
            JSON: The current JSON instance.
        """
        self.is_array = False
        return self

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the JSON object.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the JSON object.
        """
        # print('-' * 30)
        # print(self.column)
        # print('-' * 30)
        if dialect == Dialects.MSSQL:
            raise ValueError
        if dialect == Dialects.POSTGRESQL:
            x = ','.join(["'" + k + "'" + ', ' + v.build(dialect) for k, v in self.column.items()])
            if self.is_array:
                return f'json_agg(json_build_object({x}))'
            else:
                return f'json_build_object({x})'
        if dialect == Dialects.MONGO:
            ...


class Sum:
    """
    Class representing a sum operation in SQL.

    Attributes:
        column (Column): The column to sum.
    """

    def __init__(self, column):
        """
        Initializes a new instance of the Sum class.

        Args:
            column (Column): The column to sum.
        """
        self.column = column

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the sum operation.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the sum operation.
        """
        if dialect == Dialects.MSSQL:
            return f'sum({self.column._table._name}.{self.column._name})'
        if dialect == Dialects.POSTGRESQL:
            return f'sum({self.column._table._name}.{self.column._name})'
        if dialect == Dialects.MONGO:
            ...


class Query:
    """
    Class representing a SQL query.

    Attributes:
        projection (list): The columns to include in the query.
        condition (Criterion): The condition of the query.
        grouping (list): The columns to group by in the query.
        sort (list): The columns to sort by in the query.
        limit (int): The limit of the query.
        tables (list): The tables to include in the query.
        joins (list): The joins to include in the query.
        withs (list): The with clauses to include in the query.
        alias (str): The alias of the query.
    """

    def __init__(self):
        """
        Initializes a new instance of the Query class.
        """
        self.projection = []
        self.condition = None
        self.grouping = []
        self.sort = []
        self.limit = None

        self.tables = []
        self.joins = []

        self.withs = []

        self.alias = None

    def as_table(self):
        """
        Returns a Table object representing the query.

        Returns:
            Table: A Table object.
        """
        return Table(self.alias)

    def as_(self, alias):
        """
        Sets the alias of the query.

        Args:
            alias (str): The alias to set.

        Returns:
            Query: The current query instance.
        """
        self.alias = alias
        return self

    def with_(self, *query):
        """
        Adds with clauses to the query.

        Args:
            *query (Query): The queries to include in the with clause.

        Returns:
            Query: The current query instance.
        """
        self.withs = query
        return self

    def select(self, *terms):
        """
        Adds columns to the projection of the query.

        Args:
            *terms (Term): The terms to include in the projection.

        Returns:
            Query: The current query instance.
        """
        self.projection += list(terms)
        return self

    def from_(self, *tables):
        """
        Adds tables to the query.

        Args:
            *tables (Table): The tables to include in the query.

        Returns:
            Query: The current query instance.
        """
        self.tables += list(tables)
        return self

    def join(self, table, term):
        """
        Adds an inner join to the query.

        Args:
            table (Table): The table to join.
            term (Term): The condition of the join.

        Returns:
            Query: The current query instance.
        """
        self.joins.append(('inner', table, term))
        return self

    def left_join(self, table, term):
        """
        Adds a left join to the query.

        Args:
            table (Table): The table to join.
            term (Term): The condition of the join.

        Returns:
            Query: The current query instance.
        """
        self.joins.append(('left', table, term))
        return self

    def where(self, term):
        """
        Adds a condition to the query.

        Args:
            term (Term): The condition to add.

        Returns:
            Query: The current query instance.
        """
        if self.condition is None:
            self.condition = term
        else:
            self.condition = self.condition & term
        return self

    def group(self, *terms):
        """
        Adds terms to the grouping of the query.

        Args:
            *terms (Term): The terms to include in the grouping.

        Returns:
            Query: The current query instance.
        """
        self.grouping += list(terms)
        return self

    def order(self, *terms):
        """
        Adds terms to the sorting of the query.

        Args:
            *terms (Term): The terms to include in the sorting.

        Returns:
            Query: The current query instance.
        """
        self.sort += list(terms)
        return self

    def top(self, limit):
        """
        Sets the limit of the query.

        Args:
            limit (int): The limit to set.

        Returns:
            Query: The current query instance.
        """
        self.limit = limit
        return self

    def build(self, dialect, depth=0):
        """
        Builds the SQL representation of the query.

        Args:
            dialect (Dialects): The SQL dialect to use.
            depth (int, optional): The depth of the query. Defaults to 0.

        Returns:
            str: The SQL representation of the query.

        Raises:
            ValueError: If the projection or tables of the query are empty.
        """
        if dialect == Dialects.POSTGRESQL:
            sql = ''
            if len(self.withs) > 0:
                sql += f'with {",".join([x.alias + " as (" + x.build(dialect) + ")" for x in self.withs])}\n'

            if len(self.projection) == 0:
                raise ValueError('select clause cannot be empty')

            if depth >= 0:
                sql += '('
            sql += (
                    'select ' + ', '.join([x.build(dialect, depth + 1) + ' as ' + x.alias for x in self.projection]) + '\n'
            )

            if len(self.tables) == 0:
                raise ValueError('from clause cannot be empty')
            sql += 'from ' + ', '.join([x.build(dialect) for x in self.tables]) + '\n'

            if len(self.joins) > 0:
                for join in self.joins:
                    jtype, jtable, condition = join

                    sql += (
                            f'{jtype} join {jtable.build(dialect) if not isinstance(jtable, Query) else jtable.as_table().build(dialect)} on {condition.build(dialect)}'
                            + '\n'
                    )

            if self.condition is not None:
                sql += 'where ' + self.condition.build(dialect) + '\n'

            if len(self.grouping) > 0:
                sql += 'group by ' + ', '.join([x.build(dialect) for x in self.grouping]) + '\n'

            if len(self.sort) > 0:
                sql += 'order by ' + ', '.join([x.build(dialect) for (x, y) in self.sort]) + '\n'

            if self.limit is not None:
                sql += f'limit {self.limit}\n'

            if depth >= 0:
                sql += ')'

            return sql
        if dialect == Dialects.MONGO:
            # first table will be base table
            [x.build(dialect) for x in self.projection]
            [x.build(dialect) for x in self.condition]

            ret = []
            for table in self.condition:
                table_criterias = list(filter(lambda x: x.left.table == table, self.condition))
                table_selects = list(filter(lambda x: x.table == table, self.projection))
                ret.append({'$match': table_criterias[0].build(dialect)})
                ret.append({'$project': {table_selects[0].build(dialect): 1}})

            return ret
