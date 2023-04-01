import re

from joatmon.serializable import Serializable


class Relation(Serializable):
    pattern = re.compile(
        r"""(?P<collection1>.*?)
                             \.
                             (?P<field1>.*?)
                             \(\s*(?P<n1>.*?)\)
                             ->
                             (?P<collection2>.*?)
                             \.
                             (?P<field>.*?)
                             \(\s*(?P<n2>.*?)\)""", re.VERBOSE
    )

    def __init__(self, relation=''):
        super(Relation, self).__init__()

        local_collection, local_field, local_relation, foreign_collection, foreign_field, foreign_relation = Relation.pattern.match(relation).groups()

        self.local_collection = local_collection
        self.local_field = local_field
        self.local_relation = local_relation
        self.foreign_collection = foreign_collection
        self.foreign_field = foreign_field
        self.foreign_relation = foreign_relation
