from joatmon.database.constraint import UniqueConstraint
from joatmon.database.document import Document
from joatmon.database.field import Field
from joatmon.database.meta import Meta
from joatmon.database.utility import create_new_type


class Resource(Meta):
    __collection__ = '_resource'

    key = Field(str, nullable=False)
    tr = Field(str, nullable=False)
    en = Field(str, nullable=False)

    key_unique_constraint = UniqueConstraint('key')


Resource = create_new_type(Resource, (Document,))
