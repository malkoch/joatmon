import typing

from joatmon.database.document import Document
from joatmon.database.field import Field
from joatmon.database.meta import Meta
from joatmon.database.utility import create_new_type


class Log(Meta):
    __collection__ = '_log'

    level = Field(str, nullable=False)
    ip = Field(str, nullable=False)
    exception = Field(dict)
    function = Field(str, nullable=False)
    module = Field(str, nullable=False)
    language = Field(str, nullable=False)
    args = Field(tuple, nullable=False)
    kwargs = Field(dict, nullable=False)
    timed = Field(float)
    result = Field((tuple, list, dict, str))


Log = create_new_type(Log, (Document,))
