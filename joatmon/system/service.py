import datetime
import uuid

from joatmon.core.utility import new_object_id
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.module import Module


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


Service = create_new_type(Service, (Document,))


class ServiceModule(Module):
    def __init__(self, system):
        super().__init__(system)

    def create_task(self):
        ...

    def create_service(self):
        ...

    def start_task(self):
        ...

    def stop_task(self):
        ...

    def start_service(self):
        ...

    def stop_service(self):
        ...

    def list_tasks(self):
        ...

    def list_services(self):
        ...

    def get_task(self):
        ...

    def get_service(self):
        ...

    def remove_task(self):
        ...

    def remove_service(self):
        ...

    def update_task(self):
        ...

    def update_service(self):
        ...

    def run_task(self):
        ...

    def run_service(self):
        ...

    def run(self):
        ...

    def shutdown(self):
        ...
