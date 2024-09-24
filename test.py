import asyncio

from joatmon.plugin.core import register
from joatmon.plugin.database.sqlite import SQLiteDatabase
from joatmon.system.os import OS


register(SQLiteDatabase, 'sqlite', 'joatmon')

os = OS(r'/Users/malkoch/Projects/PyCharm/KORRA-Backend/korra')

# os.file_system.mkdir('test')
# os.file_system.rm('test')

# os.file_system.touch('test.json')
# os.file_system.rm('test.json')

os.file_system.write('my_script.py', '''
import os

print(os.listdir())
''')

for file_or_folder, stat in os.file_system.ls('..'):
    size = stat['size']
    unit = stat['unit']
    print(f'{size:>8} {unit:>2} {file_or_folder}')

asyncio.run(
    os.task_manager.create(
        name='greeter',
        description='greeter task',
        priority=10,
        status=True,
        mode='startup',
        script='my_script.py',
        arguments=''
    )
)
asyncio.run(
    os.job_manager.create(
        name='greeter',
        description='greeter job',
        priority=10,
        status=True,
        interval=10,
        script='my_script.py',
        arguments=''
    )
)
asyncio.run(
    os.service_manager.create(
        name='greeter',
        description='greeter service',
        priority=10,
        status=True,
        mode='automatic',
        script='my_script.py',
        arguments=''
    )
)
