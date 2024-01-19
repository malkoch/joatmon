import re

from joatmon.core.serializable import Serializable


class Relation(Serializable):
    """
    Class representing a relation in the ORM system.

    A relation represents a connection between two collections in the database.
    It is defined by a string in the format "collection1.field1(n1)->collection2.field2(n2)",
    where "n1" and "n2" are the cardinalities of the relation.

    Attributes:
        local_collection (str): The name of the local collection.
        local_field (str): The name of the local field.
        local_relation (str): The cardinality of the local relation.
        foreign_collection (str): The name of the foreign collection.
        foreign_field (str): The name of the foreign field.
        foreign_relation (str): The cardinality of the foreign relation.
    """

    pattern = re.compile(
        r"""(?P<collection1>.*?)
                             \.
                             (?P<field1>.*?)
                             \(\s*(?P<n1>.*?)\)
                             ->
                             (?P<collection2>.*?)
                             \.
                             (?P<field>.*?)
                             \(\s*(?P<n2>.*?)\)""",
        re.VERBOSE,
    )

    def __init__(self, relation=''):
        """
        Initializes a new instance of the Relation class.

        Args:
            relation (str): The string defining the relation.
        """
        super(Relation, self).__init__()

        (
            local_collection,
            local_field,
            local_relation,
            foreign_collection,
            foreign_field,
            foreign_relation,
        ) = Relation.pattern.match(relation).groups()

        self.local_collection = local_collection
        self.local_field = local_field
        self.local_relation = local_relation
        self.foreign_collection = foreign_collection
        self.foreign_field = foreign_field
        self.foreign_relation = foreign_relation
