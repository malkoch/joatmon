from enum import Enum


class Arithmetic(Enum):
    add = "+"
    sub = "-"
    mul = "*"
    div = "/"


class Comparator(Enum):
    pass


class Equality(Comparator):
    eq = "="
    ne = "<>"
    gt = ">"
    gte = ">="
    lt = "<"
    lte = "<="


class Matching(Comparator):
    not_like = " NOT LIKE "
    like = " LIKE "


class Boolean(Comparator):
    and_ = "AND"
    or_ = "OR"
    xor_ = "XOR"
    true = "TRUE"
    false = "FALSE"


class Order(Enum):
    asc = "ASC"
    desc = "DESC"


class JoinType(Enum):
    inner = ""
    left = "LEFT"
    right = "RIGHT"
    outer = "FULL OUTER"
    left_outer = "LEFT OUTER"
    right_outer = "RIGHT OUTER"
    full_outer = "FULL OUTER"
    cross = "CROSS"


class DatePart(Enum):
    year = "YEAR"
    quarter = "QUARTER"
    month = "MONTH"
    week = "WEEK"
    day = "DAY"
    hour = "HOUR"
    minute = "MINUTE"
    second = "SECOND"
    microsecond = "MICROSECOND"


class Dialects(Enum):
    MSSQL = "mssql"
    MYSQL = "mysql"
    POSTGRESQL = "postgressql"
    SQLLITE = "sqllite"
    MONGO = "mongo"


class Node:
    def __init__(self, alias):
        self._alias = alias

    def as_(self, alias):
        self._alias = alias
        return self


class Term(Node):
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

    def build(self, dialect):
        raise NotImplementedError


class ValueWrapper:
    def __init__(self, value):
        self.value = value

    def build(self, dialect):
        if self.value is None:
            return 'null'
        elif self.value is True:
            return 'true'
        elif self.value is False:
            return 'false'
        elif isinstance(self.value, str):
            return f'\'{self.value}\''
        return str(self.value)


class Criterion(Term):
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
    def __init__(self, comparator: Comparator, left: Term, right: Term, alias=None):
        super(BasicCriteria, self).__init__(alias)

        self.comparator = comparator
        self.left = left
        if right is None:
            right = ValueWrapper(right)
        if isinstance(right, (bool, int, float, str)):
            right = ValueWrapper(right)
        self.right = right

    def build(self, dialect):
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
                    return f'{self.left.build(dialect)} is {self.right.build(dialect)}'

                return f'{self.left.build(dialect)} = {self.right.build(dialect)}'
            if self.comparator == Equality.ne:
                if isinstance(self.right, ValueWrapper) and self.right.value is None:
                    return f'{self.left.build(dialect)} is not {self.right.build(dialect)}'

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
        if dialect == Dialects.MONGO:
            ret = {}
            if self.comparator == Equality.eq:
                ret[self.left.build(dialect)] = {'$eq': self.right.build(dialect)}
            print(ret)
            return ret


class ComplexCriteria(Criterion):
    def __init__(self, comparator: Comparator, left: Term, right: Term, alias=None):
        super(ComplexCriteria, self).__init__(alias)

        self.comparator = comparator
        self.left = left
        self.right = right

    def build(self, dialect):
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
                return f'{self.left.build(dialect)} and \n{self.right.build(dialect)}'
            if self.comparator == Boolean.or_:
                return f'{self.left.build(dialect)} or \n{self.right.build(dialect)}'
            return ret
        if dialect == Dialects.MONGO:
            ret = {}
            if self.comparator == Boolean.and_:
                ret['$and'] = [self.left.build(dialect), self.right.build(dialect)]
            print(ret)
            return ret


class ArithmeticExpression(Term):
    def __init__(self, operator: Arithmetic, left, right, alias=None):
        super(ArithmeticExpression, self).__init__(alias)

        self.operator = operator
        self.left = left
        if isinstance(right, (bool, int, float, str)):
            right = ValueWrapper(right)
        self.right = right

    def build(self, dialect):
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
            return ret if self._alias is None else f'{ret} as {self._alias}'
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
            return ret if self._alias is None else f'{ret} as {self._alias}'
        if dialect == Dialects.MONGO:
            ret = {}
            if self.operator == Arithmetic.sub:
                ret['$add'] = [self.left.build(dialect), self.right.build(dialect)]
            print(ret)
            return ret


class Column(Term):
    def __init__(self, name, table):
        super(Column, self).__init__(None)

        self._name = name
        self._table = table
        self._alias = None

    def build(self, dialect):
        if dialect == Dialects.POSTGRESQL:
            return f'{self._table.build(dialect)}."{self._name}"'


class Table:
    def __init__(self, name, schema):
        self._name = name
        self._schema = schema
        self._alias = None

    def __getattr__(self, item):
        return Column(item, self)

    def star(self):
        return Column('*', self)

    def as_(self, alias):
        self._alias = alias
        return self

    def build(self, dialect):
        if dialect == Dialects.POSTGRESQL:
            return f'{self._schema.build(dialect)}."{self._name}"'


class Schema:
    def __init__(self, name, database):
        self._name = name
        self._database = database
        self._alias = None

    def __getattr__(self, item):
        return Table(item, self)

    def as_(self, alias):
        self._alias = alias
        return self

    def build(self, dialect):
        if dialect == Dialects.POSTGRESQL:
            return f'{self._database.build(dialect)}."{self._name}"'


class Database:
    def __init__(self, name):
        self._name = name
        self._alias = None

    def __getattr__(self, item):
        return Schema(item, self)

    def as_(self, alias):
        self._alias = alias
        return self

    def build(self, dialect):
        if dialect == Dialects.POSTGRESQL:
            return f'"{self._name}"'


class Count:
    def __init__(self, column):
        self.column = column
        self.alias = None

    def as_(self, alias):
        self.alias = alias
        return self

    def build(self, dialect):
        if dialect == Dialects.MSSQL:
            if isinstance(self.column, Column):
                return f'count({self.column._table._name}.{self.column._name})'
            elif isinstance(self.column, Criterion):
                return f'count({self.column.build(dialect)})'
        if dialect == Dialects.POSTGRESQL:
            if isinstance(self.column, Column):
                return f'count({self.column._table._name}.{self.column._name}) as {self.alias}'
            elif isinstance(self.column, Criterion):
                return f'count(*) filter(where {self.column.build(dialect)}) as {self.alias}'
        if dialect == Dialects.MONGO:
            ...


class Sum:
    def __init__(self, column):
        self.column = column

    def build(self, dialect):
        if dialect == Dialects.MSSQL:
            return f'sum({self.column._table._name}.{self.column._name})'
        if dialect == Dialects.POSTGRESQL:
            return f'sum({self.column._table._name}.{self.column._name})'
        if dialect == Dialects.MONGO:
            ...


class Query:
    def __init__(self):
        self.projection = []
        self.condition = None
        self.grouping = []
        self.sort = []
        self.limit = None

        self.tables = []
        self.joins = []

    def select(self, *terms):
        self.projection += list(terms)
        return self

    def from_(self, *tables):
        self.tables += list(tables)
        return self

    def join(self, table, term):
        self.joins.append(('inner', table, term))
        return self

    def left_join(self, table, term):
        self.joins.append(('left', table, term))
        return self

    def where(self, term):
        if self.condition is None:
            self.condition = term
        else:
            self.condition = self.condition & term
        return self

    def group(self, *terms):
        self.grouping += list(terms)
        return self

    def order(self, *terms):
        self.sort += list(terms)
        return self

    def top(self, limit):
        self.limit = limit
        return self

    def build(self, dialect):
        # if dialect == Dialects.MSSQL:
        #     columns_in_projection = [(column._table if isinstance(column, Column) else None, column) for column in self.projection]
        #     columns_in_condition = [(column._table, column) for column in filter(lambda x: isinstance(x, Column), self.get_columns(self.condition))]
        #     columns_in_group = [(column._table, column) for column in self.grouping]
        #     columns_in_sort = [(column[0]._table, column[0], column[1]) for column in self.sort or []]

        #     tables = list(set([c[0]._name for c in columns_in_projection if c[0] is not None] + [c[0]._name for c in columns_in_condition]))

        #     select = ', '.join([x.build(dialect) for x in self.projection])
        #     group = ', '.join([f'{x[0]._name}.{x[1]._name}' for x in columns_in_group])
        #     order = ', '.join([f'{x[0]._name}.{x[1]._name} {"asc" if x[2] == 0 else "desc"}' for x in columns_in_sort])

        #     return f'select {f"top {self.limit}" if self.limit is not None else ""} {select} \n' \
        #            f'from {", ".join(tables)} \nwhere {self.condition.build(dialect)} \n' \
        #            f'group by {group} \n' \
        #            f'order by {order}'
        if dialect == Dialects.POSTGRESQL:
            sql = ''
            if len(self.projection) == 0:
                raise ValueError('select clause cannot be empty')
            sql += 'select ' + ', '.join([x.build(dialect) for x in self.projection]) + '\n'

            if len(self.tables) == 0:
                raise ValueError('from clause cannot be empty')
            sql += 'from ' + ', '.join([x.build(dialect) for x in self.tables]) + '\n'

            if len(self.joins) > 0:
                for join in self.joins:
                    jtype, jtable, condition = join

                    sql += f'{jtype} join {jtable.build(dialect)} on {condition.build(dialect)}' + '\n'

            if self.condition is not None:
                sql += 'where ' + self.condition.build(dialect)

            if len(self.grouping) > 0:
                sql += 'group by ' + ', '.join([x.build(dialect) for x in self.grouping]) + '\n'

            if len(self.sort) > 0:
                sql += 'order by ' + ', '.join([x.build(dialect) for (x, y) in self.sort]) + '\n'

            return sql
        if dialect == Dialects.MONGO:
            # first table will be base table
            select = [x.build(dialect) for x in self.projection]
            where = [x.build(dialect) for x in self.condition]

            ret = []
            for table in self.condition:
                table_criterias = list(filter(lambda x: x.left.table == table, self.condition))
                table_selects = list(filter(lambda x: x.table == table, self.projection))
                ret.append({'$match': table_criterias[0].build(dialect)})
                ret.append({'$project': {table_selects[0].build(dialect): 1}})

            return ret
