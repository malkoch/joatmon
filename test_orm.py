from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import create_new_type, Document
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta


class Resource(Meta):
    __collection__ = '_resource'

    key = Field(str, nullable=False)
    tr = Field(str, nullable=False)
    en = Field(str, nullable=False)

    key_unique_constraint = UniqueConstraint('key')


Resource = create_new_type(Resource, (Document,))

resource = Resource(**{'key': '1', 'tr': 'bir', 'en': 'one'})
dictionary = resource.validate()
print(resource)
print(dictionary)
