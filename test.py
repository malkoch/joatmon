from joatmon.plugin.core import register
from joatmon.plugin.database.sqlite import SQLiteDatabase
from joatmon.system.fs import FSModule
from joatmon.system.os import OS


register(SQLiteDatabase, 'sqlite', 'joatmon')

os = OS()
os.inject('fs', FSModule(r'/Volumes/T7 Touch/IVA'))

os.fs.mkdir('test')
os.fs.rm('test')

os.fs.touch('test.json')
os.fs.rm('test.json')

for file_or_folder, stat in os.fs.ls('..'):
    size = stat['size']
    unit = stat['unit']
    print(f'{size:>8} {unit:>2} {file_or_folder}')
