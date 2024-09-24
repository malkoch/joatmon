import datetime
import uuid

from joatmon.core.exception import CoreException
from joatmon.core.utility import new_object_id, to_list_async
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.module import Module


class TaskException(CoreException):
    ...


class Task(Meta):
    __collection__ = 'task'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    name = Field(str, nullable=False, default='')
    description = Field(str, nullable=False, default='')
    priority = Field(int, nullable=False, default=10)
    status = Field(bool, nullable=False, default=True)
    mode = Field(str, nullable=False, default='manual')
    interval = Field(int, nullable=True)
    script = Field(str, nullable=False)
    arguments = Field(str, nullable=False, default='')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    last_run_time = Field(datetime.datetime, nullable=True)
    next_run_time = Field(datetime.datetime, nullable=True)


Task = create_new_type(Task, (Document,))


class TaskModule(Module):
    def __init__(self, system):
        super().__init__(system)

    async def create(self, name, description, priority, status, mode, interval, script: str, arguments):
        await self.system.persistence.drop(Task)

        script = self.system.file_system._get_host_path(script)

        if not self.system.file_system.exists(script):
            raise TaskException(f'{self.system.file_system._get_system_path(script)} does not exist')
        if not self.system.file_system.isfile(script):
            raise TaskException(f'{self.system.file_system._get_system_path(script)} is not a file')

        if mode not in ['manual', 'interval', 'startup', 'shutdown']:
            raise TaskException(f'{mode} is not a valid mode')

        if mode in ['interval'] and (interval is None or interval <= 0):
            interval = 60 * 60 * 24
        else:
            interval = None

        await self.system.persistence.insert(
            Task, {
                'name': name,
                'description': description,
                'priority': priority,
                'status': status,
                'mode': mode,
                'interval': interval,
                'script': script,
                'arguments': arguments,
            }
        )
        ret = await to_list_async(self.system.persistence.read(Task, {}))
        print(ret)

    def start(self):
        ...

    def stop(self):
        ...

    def list(self):
        ...

    def get(self):
        ...

    def remove(self):
        ...

    def update(self):
        ...

    def run(self):
        ...

    def shutdown(self):
        ...
