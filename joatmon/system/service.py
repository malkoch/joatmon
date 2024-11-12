import datetime
import importlib.util
import inspect
import json
import os
import sys
import uuid

from joatmon.core.event import AsyncEvent
from joatmon.core.exception import CoreException
from joatmon.core.utility import first_async, new_object_id, to_list_async
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.module import Module


class ServiceException(CoreException):
    ...


class BaseService:
    def __init__(self, system):
        ...

    @staticmethod
    def get_cls(path):
        module_name = os.path.abspath(path).split(os.sep)[-1].split('.')[0]

        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        classes = inspect.getmembers(module, inspect.isclass)
        cls = list(filter(lambda x: issubclass(x[1], BaseService) and x[1] is not BaseService, classes))[0][1]
        return cls

    async def run(self, *args, **kwargs):
        raise NotImplementedError


class Service(Meta):
    __collection__ = 'service'

    structured = True
    force = True

    id = Field(uuid.UUID, nullable=False, default=new_object_id, primary=True)
    name = Field(str, nullable=False, default='')
    description = Field(str, nullable=False, default='')
    priority = Field(int, nullable=False, default=10)
    mode = Field(str, nullable=False, default='manual')
    retry = Field(int, nullable=True)
    script = Field(str, nullable=False)
    args = Field(str, nullable=False, default='[]')
    kwargs = Field(str, nullable=False, default='{}')
    created_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)
    updated_at = Field(datetime.datetime, nullable=False, default=datetime.datetime.now)


Service = create_new_type(Service, (Document,))


class ServiceModule(Module):
    def __init__(self, system):
        super().__init__(system)

        self.events = {
            'on_start': AsyncEvent(),
            'on_end': AsyncEvent(),
            'on_error': AsyncEvent()
        }

    async def _on_start(self, service):
        ...

    async def _on_end(self, service):
        ...

    async def _on_error(self, service):
        if service.retry:
            await self.run(service.id)

    async def create(self, name, description, priority, mode, script: str, *args, **kwargs):
        service = await first_async(self.system.persistence.read(Service, {'name': name}))

        script = self.system.file_system._get_host_path(script)

        if not self.system.file_system.exists(script):
            raise ServiceException(f'{self.system.file_system._get_system_path(script)} does not exist')
        if not self.system.file_system.isfile(script):
            raise ServiceException(f'{self.system.file_system._get_system_path(script)} is not a file')

        if mode not in ['manual', 'automatic']:
            raise ServiceException(f'{mode} is not a valid mode')

        if service:
            await self.system.persistence.update(
                Service, {'id': service.id}, {
                    'name': name,
                    'description': description,
                    'priority': priority,
                    'mode': mode,
                    'script': script,
                    'args': json.dumps(args),
                    'kwargs': json.dumps(kwargs)
                }
            )
        else:
            await self.system.persistence.insert(
                Service, {
                    'name': name,
                    'description': description,
                    'priority': priority,
                    'mode': mode,
                    'script': script,
                    'args': json.dumps(args),
                    'kwargs': json.dumps(kwargs)
                }
            )

    async def run(self, object_id):
        service = await first_async(self.system.persistence.read(Service, {'id': object_id}))
        await self.system.process_manager.run(service)

    async def stop(self, object_id):
        ...

    async def list(self):
        return await to_list_async(self.system.persistence.read(Service, {}))

    async def get(self, object_id):
        return await first_async(self.system.persistence.read(Service, {'id': object_id}))

    async def remove(self, object_id):
        await self.system.persistence.delete(Service, {'id': object_id})

    async def update(self, object_id):
        ...

    async def start(self):
        self.events['on_start'] += self._on_start
        self.events['on_end'] += self._on_end
        self.events['on_error'] += self._on_error

        services = await self.list()
        for service in filter(lambda x: x.mode == 'automatic', services):
            await self.run(service.id)

    async def shutdown(self):
        self.events['on_start'] -= self._on_start
        self.events['on_end'] -= self._on_end
        self.events['on_error'] -= self._on_error
