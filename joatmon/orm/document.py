from joatmon.orm.field import get_converter
from joatmon.orm.meta import Meta
from joatmon.core.serializable import Serializable


class Document(Serializable):  # need to have copy and deepcopy functions as well
    """
    Base class for all documents in the ORM system.

    Attributes:
        __metaclass__ (Meta): The metaclass that contains the document's metadata.
    """

    __metaclass__ = Meta

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Document class.

        Args:
            **kwargs: The initial values for the document's fields.
        """
        super(Document, self).__init__(**kwargs)

        for name, field in self.__metaclass__.fields(self.__metaclass__).items():
            if name not in kwargs:
                setattr(self, name, None)

    def __getattr__(self, item):
        """
        Gets the value of a field.

        Args:
            item (str): The name of the field.

        Returns:
            The value of the field, or None if the field does not exist.
        """
        return self.__dict__.get(item, None)

    def __setattr__(self, key, value):
        """
        Sets the value of a field.

        Args:
            key (str): The name of the field.
            value: The new value for the field.
        """
        # if value is callable, get the value first
        # then use converter to get the value

        if key not in self.__metaclass__.fields(self.__metaclass__).keys():
            self.__dict__[key] = value
            return

        field = self.__metaclass__.fields(self.__metaclass__)[key]
        # self.__dict__[key] = get_converter(field.dtype)(value)
        self.__dict__[key] = get_converter(field)(value)

    def __len__(self):
        """
        Gets the number of fields in the document.

        Returns:
            int: The number of fields in the document.
        """
        return len(self.__dict__.keys())

    def __iter__(self):
        """
        Gets an iterator for the fields in the document.

        Returns:
            iterator: An iterator for the fields in the document.
        """
        for k in self.__dict__.keys():
            yield k, getattr(self, k, None)

    def __getitem__(self, item):
        """
        Gets the value of a field.

        Args:
            item (str): The name of the field.

        Returns:
            The value of the field, or None if the field does not exist.
        """
        return getattr(self, item, None)

    def keys(self):
        """
        Gets the names of the fields in the document.

        Returns:
            list: The names of the fields in the document.
        """
        return list(self.__dict__.keys())

    def values(self):
        """
        Gets the values of the fields in the document.

        Returns:
            list: The values of the fields in the document.
        """
        return list(map(lambda x: getattr(self, x, None), self.__dict__.keys()))

    def validate(self):
        """
        Validates the document.

        Returns:
            dict: A dictionary containing the validated fields and their values.

        Raises:
            ValueError: If the document is not valid.
        """
        # if meta is not structured and not forced, might still need to soft validate
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
                if ((value is not None and field.nullable) or not field.nullable) and not isinstance(
                        value, field.dtype
                ):
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
    """
    Creates a new type with the specified metaclass and subclasses.

    Args:
        meta (Meta): The metaclass for the new type.
        subclasses (tuple): The subclasses for the new type.

    Returns:
        type: The new type.
    """
    return type(meta.__collection__, subclasses, {'__metaclass__': meta})
