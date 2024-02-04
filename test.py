import asyncio

from joatmon.plugin.core import register
from joatmon.plugin.database.sqlite import SQLiteDatabase
from joatmon.system.os import OS


register(SQLiteDatabase, 'sqlite', 'joatmon')

os = OS('.')
os.touch('test.txt')

asyncio.run(os.test())
asyncio.run(
    os.create_task(
        'test',
        'test',
        10,
        True,
        'startup',
        None,
        'greet',
        ''
    )
)
