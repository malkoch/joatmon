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


class ForeignKey(Key):
    """
    Constraint that checks whether a field's value is a valid foreign key.
    """

    def __init__(self, relation: str):
        super(ForeignKey, self).__init__()

        self.relation = relation

    @property
    def from_table(self):
        return self.relation.split('->')[0].split('.')[0]

    @property
    def from_field(self):
        return self.relation.split('->')[0].split('.')[1]

    @property
    def to_table(self):
        return self.relation.split('->')[1].split('.')[0]

    @property
    def to_field(self):
        return self.relation.split('->')[1].split('.')[1]
