from enum import Enum


class Arithmetic(Enum):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    add = '+'
    sub = '-'
    mul = '*'
    div = '/'


class Comparator(Enum):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """


class Equality(Comparator):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    eq = '='
    ne = '<>'
    gt = '>'
    gte = '>='
    lt = '<'
    lte = '<='


class Matching(Comparator):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    not_like = ' NOT LIKE '
    like = ' LIKE '


class Boolean(Comparator):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    and_ = 'AND'
    or_ = 'OR'
    xor_ = 'XOR'
    true = 'TRUE'
    false = 'FALSE'


class Order(Enum):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    asc = 'ASC'
    desc = 'DESC'


class JoinType(Enum):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    MSSQL = 'mssql'
    MYSQL = 'mysql'
    POSTGRESQL = 'postgressql'
    SQLLITE = 'sqllite'
    MONGO = 'mongo'


class Node:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, alias):
        self.alias = alias

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self


class Term(Node):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        raise NotImplementedError


class ValueWrapper:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, value):
        self.value = value

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __and__(self, other):
        return ComplexCriteria(Boolean.and_, self, other)

    def __or__(self, other):
        return ComplexCriteria(Boolean.or_, self, other)

    def __xor__(self, other):
        ...

    @staticmethod
    def any():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...

    @staticmethod
    def all():
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ...


class BasicCriteria(Criterion):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, comparator: Comparator, left: Term, right: Term, alias=None):
        super(ComplexCriteria, self).__init__(alias)

        self.comparator = comparator
        self.left = left
        self.right = right

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, name, table):
        super(Column, self).__init__(None)

        self._name = name
        self._table = table
        self.alias = None

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if dialect == Dialects.POSTGRESQL:
            return f'{self._table.build(dialect)}."{self._name}"'


class Table:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, name, schema=None):
        self._name = name
        self._schema = schema
        self.alias = None

    def __getattr__(self, item):
        return Column(item, self)

    def star(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return Column('*', self)

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if dialect == Dialects.POSTGRESQL:
            if self._schema is None:
                return f'"{self._name}"'
            else:
                return f'{self._schema.build(dialect)}."{self._name}"'


class Schema:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, name, database):
        self._name = name
        self._database = database
        self.alias = None

    def __getattr__(self, item):
        return Table(item, self)

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if dialect == Dialects.POSTGRESQL:
            return f'{self._database.build(dialect)}."{self._name}"'


class Database:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, name):
        self._name = name
        self.alias = None

    def __getattr__(self, item):
        return Schema(item, self)

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if dialect == Dialects.POSTGRESQL:
            return f'"{self._name}"'


class Count:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, column):
        self.column = column
        self.alias = None

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, *column):
        self.column = column
        self.alias = None

        self.dtype = None

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self

    def as_text(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.dtype = 'text'
        return self

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, **kwargs):
        self.column = kwargs
        self.alias = None

        self.dtype = None
        self.is_array = True

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self

    def as_text(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.dtype = 'text'
        return self

    def array(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.is_array = True
        return self

    def object(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.is_array = False
        return self

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, column):
        self.column = column

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if dialect == Dialects.MSSQL:
            return f'sum({self.column._table._name}.{self.column._name})'
        if dialect == Dialects.POSTGRESQL:
            return f'sum({self.column._table._name}.{self.column._name})'
        if dialect == Dialects.MONGO:
            ...


class Query:
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self):
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        return Table(self.alias)

    def as_(self, alias):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.alias = alias
        return self

    def with_(self, *query):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.withs = query
        return self

    def select(self, *terms):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.projection += list(terms)
        return self

    def from_(self, *tables):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.tables += list(tables)
        return self

    def join(self, table, term):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.joins.append(('inner', table, term))
        return self

    def left_join(self, table, term):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.joins.append(('left', table, term))
        return self

    def where(self, term):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if self.condition is None:
            self.condition = term
        else:
            self.condition = self.condition & term
        return self

    def group(self, *terms):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.grouping += list(terms)
        return self

    def order(self, *terms):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.sort += list(terms)
        return self

    def top(self, limit):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.limit = limit
        return self

    def build(self, dialect, depth=0):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
