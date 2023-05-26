import uuid

from joatmon.orm.document import (
    create_new_type,
    Document
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta


class T1(Meta):
    object_id = Field(uuid.UUID)
    name = Field(str, resource='{{@resource.{0}}}')
    description = Field(str, resource='{{@resource.{0}}}')


class T2(Meta):
    object_id = Field(uuid.UUID)
    type = Field(
        object, fields={
            'name': Field(str, resource='{{@resource.{0}}}'),
            'description': Field(str, resource='{{@resource.{0}}}')
        }
    )


class T3(Meta):
    object_id = Field(uuid.UUID)
    slots = Field(
        list, fields={
            'item': Field(
                object, fields={
                    'name': Field(str, resource='{{@resource.{0}}}'),
                    'description': Field(str, resource='{{@resource.{0}}}')
                }
            )
        }
    )


T1 = create_new_type(T1, (Document,))
T2 = create_new_type(T2, (Document,))
T3 = create_new_type(T3, (Document,))

t1 = T1(**{
    'object_id': uuid.uuid4(),
    'name': '123',
    'description': '123'
})
print(t1)

t2 = T2(**{
    'object_id': uuid.uuid4(),
    'type': {
        'name': '123',
        'description': '123'
    }
})
print(t2)

t3 = T3(**{
    'object_id': uuid.uuid4(),
    'slots': [
        {
            'item': {
                'name': '123',
                'description': '123'
            }
        },
        {
            'item': {
                'name': '456',
                'description': '456'
            }
        }
    ]
})
print(t3)
