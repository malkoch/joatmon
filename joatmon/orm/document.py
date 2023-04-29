from joatmon.serializable import Serializable
from joatmon.utility import get_converter
from joatmon.orm.meta import Meta


class Document(Serializable):  # need to have copy and deepcopy functions as well
    __metaclass__ = Meta

    def __init__(self, **kwargs):
        super(Document, self).__init__(**kwargs)

        for name, field in self.__metaclass__.fields(self.__metaclass__).items():
            if name not in kwargs:
                setattr(self, name, None)

    def __getattr__(self, item):
        return self.__dict__.get(item, None)

    def __setattr__(self, key, value):
        if key not in self.__metaclass__.fields(self.__metaclass__).keys():
            self.__dict__[key] = value
            return

        field = self.__metaclass__.fields(self.__metaclass__)[key]
        self.__dict__[key] = get_converter(field.dtype)(value)

    def __len__(self):
        return len(self.__dict__.keys())

    def __iter__(self):
        for k in self.__dict__.keys():
            yield k, getattr(self, k, None)

    def __getitem__(self, item):
        return getattr(self, item, None)

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(map(lambda x: getattr(self, x, None), self.__dict__.keys()))

    def validate(self):
        if not self.__metaclass__.structured and not self.__metaclass__.force:
            raise ValueError('unstructured document cannot be validated')

        ret = {}
        for name, field in self.__metaclass__.fields(self.__metaclass__).items():
            value = getattr(self, name, None)

            default_value = field.default()

            if value is None and not field.nullable:
                setattr(self, name, default_value)

            value = getattr(self, name, None)

            if value is None and not field.nullable:
                raise ValueError(f'field {name} is not nullable')

            if isinstance(field.dtype, (tuple, list)):
                if ((value is not None and field.nullable) or not field.nullable) and not isinstance(value,field.dtype):
                    raise ValueError(
                        f'field {name} has to be one of the following {field.dtype} not {type(value).__name__}'
                    )
            else:
                if ((value is not None and field.nullable) or not field.nullable) and type(value) is not field.dtype:
                    raise ValueError(f'field {name} has to be type {field.dtype} not {type(value).__name__}')

            constraints = self.__metaclass__.constraints(self.__metaclass__).values()
            constraints = list(filter(lambda x: x.field == name, constraints))

            if not field.nullable and constraints is not None:
                for constraint in constraints:
                    constraint.check(value)

            ret[name] = getattr(self, name, None)

        return ret


def create_new_type(meta, subclasses):
    return type(meta.__collection__, subclasses, {'__metaclass__': meta})
