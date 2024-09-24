import datetime
import uuid

from joatmon.core.exception import CoreException
from joatmon.core.utility import new_object_id, to_list_async
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.module import Module


class ServiceException(CoreException):
    ...


class Service(Meta):
    __collection__ = 'service'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    name = Field(str, nullable=False, default='')
    description = Field(str, nullable=False, default='')
    priority = Field(int, nullable=False, default=10)
    status = Field(bool, nullable=False, default=True)
    mode = Field(str, nullable=False, default='manual')
    retry = Field(int, nullable=True)
    script = Field(str, nullable=False)
    arguments = Field(str, nullable=False, default='')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    last_run_time = Field(datetime.datetime, nullable=True)
    next_run_time = Field(datetime.datetime, nullable=True)
    is_deleted = Field(bool, nullable=False, default=False)


Service = create_new_type(Service, (Document,))


class ServiceModule(Module):
    def __init__(self, system):
        super().__init__(system)

    async def create(self, name, description, priority, status, mode, script: str, arguments):
        await self.system.persistence.drop(Service)

        script = self.system.file_system._get_host_path(script)

        if not self.system.file_system.exists(script):
            raise ServiceException(f'{self.system.file_system._get_system_path(script)} does not exist')
        if not self.system.file_system.isfile(script):
            raise ServiceException(f'{self.system.file_system._get_system_path(script)} is not a file')

        if mode not in ['manual', 'automatic']:
            raise ServiceException(f'{mode} is not a valid mode')

        await self.system.persistence.insert(
            Service, {
                'name': name,
                'description': description,
                'priority': priority,
                'status': status,
                'mode': mode,
                'script': script,
                'arguments': arguments,
            }
        )

    async def run(self, object_id):
        ...

    async def stop(self, object_id):
        ...

    async def list(self):
        ret = await to_list_async(self.system.persistence.read(Service, {'is_deleted': False}))
        print(ret)

    async def get(self, object_id):
        ...

    async def remove(self, object_id):
        ...

    async def update(self, object_id):
        ...

    async def start(self):
        await self.list()

    async def shutdown(self):
        ...
