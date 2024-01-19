from joatmon.core.serializable import Serializable


class Index(Serializable):
    """
    Class representing an index in the ORM system.

    An index is a data structure that improves the speed of data retrieval operations on a database table.
    It provides a direct path to the data, which can be used to locate data without having to search every row.

    Attributes:
        field (str): The field on which the index is created.
    """

    def __init__(self, field):
        """
        Initializes a new instance of the Index class.

        Args:
            field (str): The field on which the index is created.
        """
        super(Index, self).__init__()

        self.field = field
