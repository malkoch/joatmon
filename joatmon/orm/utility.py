import re
import uuid
from datetime import datetime

from joatmon.utility import get_converter

email_pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
ip_address_pattern = re.compile(r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                                r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                                r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\."
                                r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")


def empty_object_id():
    return uuid.UUID('00000000-0000-0000-0000-000000000000')


def new_object_id():
    return uuid.uuid4()


def current_time():
    return datetime.now()


def new_nickname():
    return f'random_nickname_{uuid.uuid4()}'


def new_password():
    return f'random_password_{uuid.uuid4()}'


def mail_validator(email):
    return email_pattern.match(email) is not None


def ip_validator(ip):
    return ip_address_pattern.match(ip) is not None


def normalize_kwargs(meta, **kwargs):
    ret = {}

    fields = meta.fields(meta)
    for key in kwargs.keys():
        field = list(filter(lambda x: x[0] == key, fields.items()))
        if len(field) != 1:
            raise ValueError(f'field {key} has to be only one on the document')
        field = field[0][1]

        ret[key] = get_converter(field.kind)(kwargs[key])
    return ret


def create_new_type(meta, subclasses):
    return type(meta.__collection__, subclasses, {'__metaclass__': meta})
