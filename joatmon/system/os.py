import asyncio

from joatmon.core.exception import CoreException
from joatmon.system.fs import FileSystemModule
from joatmon.system.job import JobModule
from joatmon.system.module import ModuleManager
from joatmon.system.persistence import PersistenceModule
from joatmon.system.process import ProcessModule
from joatmon.system.service import ServiceModule
from joatmon.system.task import TaskModule


class OSException(CoreException):
    ...


class OS:
    def __init__(self, path: str):
        self.mm = ModuleManager()

        self.inject('file_system', FileSystemModule(self, path))
        self.inject('process_manager', ProcessModule(self))
        self.inject('task_manager', TaskModule(self))
        self.inject('job_manager', JobModule(self))
        self.inject('service_manager', ServiceModule(self))
        self.inject('persistence', PersistenceModule(self, 'sqlite'))

        self.waiter = asyncio.Event()

    def inject(self, name, module):
        self.mm.register(name, module)

    def __getattr__(self, item):
        return self.mm.__getattr__(item)

    async def start(self):
        await self.process_manager.start()
        await self.task_manager.start()
        await self.job_manager.start()
        await self.service_manager.start()

    async def shutdown(self):
        await self.service_manager.shutdown()
        await self.job_manager.shutdown()
        await self.task_manager.shutdown()
        await self.process_manager.shutdown()

        self.waiter.set()
