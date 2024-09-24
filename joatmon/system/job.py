import datetime
import uuid

from joatmon.core.utility import new_object_id
from joatmon.orm.document import Document, create_new_type
from joatmon.orm.field import Field
from joatmon.orm.meta import Meta
from joatmon.system.module import Module


class Job(Meta):
    __collection__ = 'job'

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


Job = create_new_type(Job, (Document,))


class JobModule(Module):
    def __init__(self, system):
        super().__init__(system)

    def create(self):
        ...

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
