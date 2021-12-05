import typing

from joatmon.serializable import Serializable


class Response(Serializable):
    def __init__(self, data: typing.Optional[typing.Union[dict, list, Serializable]] = None, success: bool = False, message: typing.Optional[str] = None):
        super(Response, self).__init__(data=data, success=success, message=message)
