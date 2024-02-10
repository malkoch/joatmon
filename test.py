import asyncio

from joatmon.plugin.core import register
from joatmon.plugin.database.sqlite import SQLiteDatabase
from joatmon.system.os import OS


register(SQLiteDatabase, 'sqlite', 'joatmon')

os = OS('.')
# os.touch('test.json')
os.ls('..')

asyncio.run(
    os.create_task(
        'startup greeter',
        'a task to greet on startup',
        10,
        True,
        'startup',
        None,
        'greet',
        ''
    )
)
