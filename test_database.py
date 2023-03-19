import asyncio

from joatmon import context
from joatmon.orm.document import (
    create_new_type,
    Document
)
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.plugin.core import register
from joatmon.plugin.database.couchbase import CouchBaseDatabase
from joatmon.plugin.database.elastic import ElasticDatabase
from joatmon.plugin.web.core import UserPlugin


class Log(Meta):
    __collection__ = 'log'

    level = Field(str, nullable=False)
    ip = Field(str, nullable=False)
    exception = Field(dict)
    function = Field(str, nullable=False)
    module = Field(str, nullable=False)
    language = Field(str, nullable=False)
    args = Field(tuple, nullable=False)
    kwargs = Field(dict, nullable=False)
    timed = Field(float)
    result = Field((tuple, list, dict, str))


Log = create_new_type(Log, (Document,))


class CTX:
    ...


c = CTX()
context.set_ctx(c)

register(UserPlugin, 'user_plugin', 'web')
# register(ElasticDatabase, 'elastic', 'http://localhost:9200', 'user_plugin')
register(CouchBaseDatabase, 'couchbase', 'couchbase://localhost', 'ToG', '_default', 'malkoch', 'malkoch', 'user_plugin')


async def test():
    e = context.get_value('couchbase')
    await e.insert(
        Log(
            **{
                'level': 'debug',
                'ip': '0.0.0.0',
                'exception': None,
                'function': 'test',
                'module': 'main',
                'language': 'en',
                'args': (),
                'kwargs': {},
                'timed': None,
                'result': {}
            }
        )
    )


asyncio.run(test())
