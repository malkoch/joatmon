import enum

from joatmon.core.serializable import Serializable


class Key(Serializable):
    """
    Class representing an index in the ORM system.

    An index is a data structure that improves the speed of data retrieval operations on a database table.
    It provides a direct path to the data, which can be used to locate data without having to search every row.
    """

    def __init__(self):
        """
        Initializes a new instance of the Index class.
        """
        super(Key, self).__init__()


class PrimaryKey(Key):
    """
    Constraint that checks whether a field's value is a valid primary key.

    Attributes:
        field (str): The field on which the index is created.
    """

    def __init__(self, field):
        super(PrimaryKey, self).__init__()

        self.field = field


class ForeignKeyType(enum.Enum):
    OneToOne = 1
    OneToMany = 2
    ManyToOne = 3
    ManyToMany = 4


class ForeignKey(Key):
    """
    Constraint that checks whether a field's value is a valid foreign key.
    """

    def __enter__(self, from_table, from_field, to_table, to_field, rel: ForeignKeyType = ForeignKeyType.OneToOne):
        super(ForeignKey, self).__init__()

        self.from_table = from_table
        self.from_field = from_field
        self.to_table = to_table
        self.to_field = to_field
        self.rel = rel
