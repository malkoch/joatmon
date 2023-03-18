import asyncio
from datetime import datetime
from uuid import UUID

from joatmon import context
from joatmon.core.utility import (
    current_time,
    empty_object_id,
    first_async,
    new_object_id
)
from joatmon.orm.constraint import UniqueConstraint
from joatmon.orm.document import (
    create_new_type,
    Document
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.plugin.core import register
from joatmon.plugin.database.couchbase import CouchBaseDatabase
from joatmon.plugin.database.elastic import ElasticDatabase
from joatmon.plugin.database.mongo import MongoDatabase
from joatmon.plugin.database.postgresql import PostgreSQLDatabase


class MyMeta(Meta):
    structured = True

    object_id = Field(UUID, nullable=False, default=new_object_id, primary=True)
    created_at = Field(datetime, nullable=False, default=current_time)
    creator_id = Field(UUID, nullable=False, default=empty_object_id)
    updated_at = Field(datetime, nullable=False, default=current_time)
    updater_id = Field(UUID, nullable=False, default=empty_object_id)
    deleted_at = Field(datetime, nullable=True, default=current_time)
    deleter_id = Field(UUID, nullable=True, default=empty_object_id)
    is_deleted = Field(bool, nullable=False, default=False)

    unique_constraint_object_id = UniqueConstraint('object_id')


class MyMeta2(Meta):
    structured = False


class StructuredLog(MyMeta):
    __collection__ = 'log'

    level = Field(str, nullable=False)
    ip = Field(str, nullable=False)
    exception = Field(str)
    function = Field(str, nullable=False)
    module = Field(str, nullable=False)
    language = Field(str, nullable=False)
    args = Field(str, nullable=False)
    kwargs = Field(str, nullable=False)
    timed = Field(float)
    result = Field(str)


class UnStructuredLog(MyMeta2):
    __collection__ = 'log'


StructuredLog = create_new_type(StructuredLog, (Document,))
UnStructuredLog = create_new_type(UnStructuredLog, (Document,))

sl = StructuredLog(**{'level': '1', 'ip': '1', 'function': '', 'module': '', 'language': '', 'args': '()', 'kwargs': '{}'})
ul = UnStructuredLog(**{'object_id': new_object_id(), 'level': '1', 'ip': '1', 'function': '', 'module': '', 'language': '', 'args': (), 'kwargs': {}})


class CTX:
    ...


c = CTX()

context.set_ctx(c)

register(MongoDatabase, 'mongo', 'mongodb://malkoch:malkoch@127.0.0.1:27017/?replicaSet=rs0', 'ToG')
register(ElasticDatabase, 'elastic', 'http://localhost:9200')
register(CouchBaseDatabase, 'couchbase', 'couchbase://localhost', 'ToG', '_default', 'malkoch', 'malkoch')
register(PostgreSQLDatabase, 'postgresql', '127.0.0.1', 5432, 'malkoch', 'malkoch', 'ToG')


async def test():
    # document = await first_async(context.get_value('mongo').insert(UnStructuredLog, dict(**ul)))
    # document = await first_async(context.get_value('mongo').read(UnStructuredLog, {'object_id': document.object_id}))
    # document.level = '2'
    # await context.get_value('mongo').update(UnStructuredLog, {'object_id': document.object_id}, document)
    # document = await first_async(context.get_value('mongo').read(UnStructuredLog, {'object_id': document.object_id}))
    # await context.get_value('mongo').delete(UnStructuredLog, {'object_id': document.object_id})

    # document = await first_async(context.get_value('elastic').insert(UnStructuredLog, dict(**ul)))
    # document = await first_async(context.get_value('elastic').read(UnStructuredLog, {'object_id': document.object_id}))
    # document.level = '2'
    # await context.get_value('elastic').update(UnStructuredLog, {'object_id': document.object_id}, document)
    # document = await first_async(context.get_value('elastic').read(UnStructuredLog, {'object_id': document.object_id}))
    # await context.get_value('elastic').delete(UnStructuredLog, {'object_id': document.object_id})

    document = await first_async(context.get_value('couchbase').insert(UnStructuredLog, dict(**ul)))
    document = await first_async(context.get_value('couchbase').read(UnStructuredLog, {'object_id': document.object_id}))
    document.level = '2'
    await context.get_value('couchbase').update(UnStructuredLog, {'object_id': document.object_id}, document)
    document = await first_async(context.get_value('couchbase').read(UnStructuredLog, {'object_id': document.object_id}))
    await context.get_value('couchbase').delete(UnStructuredLog, {'object_id': document.object_id})

    # document = await first_async(context.get_value('postgresql').insert(StructuredLog, dict(**sl)))
    # document = await first_async(context.get_value('postgresql').read(StructuredLog, {'object_id': document.object_id}))
    # document.level = '2'
    # await context.get_value('postgresql').update(StructuredLog, {'object_id': document.object_id}, document)
    # document = await first_async(context.get_value('postgresql').read(StructuredLog, {'object_id': document.object_id}))
    # await context.get_value('postgresql').delete(StructuredLog, {'object_id': document.object_id})
    # document = await first_async(context.get_value('postgresql').read(StructuredLog, {'object_id': document.object_id}))

    # # await context.get_value('postgresql').delete(StructuredLog, {})
    # async for document in context.get_value('postgresql').read(StructuredLog, {}):
    #     print(document)


asyncio.run(test())
