from joatmon.core.serializable import Serializable


class ValidationException(Exception):
    """
    Exception raised for errors in the validation process.

    Attributes:
        message -- explanation of the error
    """


class Constraint(Serializable):
    """
    Base class for all constraints.

    Attributes:
        field (str): The field to which the constraint applies.
        validator (callable): A function that validates the field's value.
    """

    def __init__(self, field, validator=None):
        super(Constraint, self).__init__()

        self.field = field
        self.validator = validator

    @staticmethod
    def create(constraint_type, **kwargs):
        """
        Factory method for creating constraints.

        Args:
            constraint_type (str): The type of constraint to create.
            **kwargs: Additional keyword arguments for the constraint's constructor.

        Returns:
            Constraint: A new constraint of the specified type.
        """
        if constraint_type == 'length':
            return LengthConstraint(**kwargs)
        if constraint_type == 'integer':
            return IntegerValueConstraint(**kwargs)
        if constraint_type == 'unique':
            return UniqueConstraint(**kwargs)
        if constraint_type == 'custom':
            return CustomConstraint(**kwargs)

    def check(self, obj):
        """
        Checks whether the constraint is satisfied.

        Args:
            obj: The object to check.

        Returns:
            bool: True if the constraint is satisfied, False otherwise.

        Raises:
            ValidationException: If the constraint is not satisfied.
        """
        if self.validator is not None:
            if callable(self.validator):
                if self.validator(obj):
                    return True
                else:
                    raise ValidationException(f'field.{self.field}.not_valid')
            else:
                return True
        else:
            return True


class LengthConstraint(Constraint):
    """
    Constraint that checks whether a field's value has a valid length.

    Attributes:
        min_length (int): The minimum valid length. None if there is no minimum.
        max_length (int): The maximum valid length. None if there is no maximum.
    """

    def __init__(self, field, min_length=None, max_length=None):
        if min_length is None and max_length is None:
            raise ValueError('at least one of the min_length, max_length has to be provided')

        if min_length is None:
            validator = lambda x: len(x) <= max_length
        elif max_length is None:
            validator = lambda x: min_length <= len(x)
        else:
            validator = lambda x: min_length <= len(x) <= max_length

        super(LengthConstraint, self).__init__(field, validator)

        self.min_length = min_length
        self.max_length = max_length


class IntegerValueConstraint(Constraint):
    """
    Constraint that checks whether a field's value is within a valid range.

    Attributes:
        min_value (int): The minimum valid value. None if there is no minimum.
        max_value (int): The maximum valid value. None if there is no maximum.
    """

    def __init__(self, field, min_value=None, max_value=None):
        if min_value is None and max_value is None:
            raise ValueError('at least one of the min_value, max_value has to be provided')

        if min_value is None:
            validator = lambda x: x <= max_value
        elif max_value is None:
            validator = lambda x: min_value <= x
        else:
            validator = lambda x: min_value <= x <= max_value

        super(IntegerValueConstraint, self).__init__(field, validator)

        self.min_value = min_value
        self.max_value = max_value


class PrimaryKeyConstraint(Constraint):
    """
    Constraint that checks whether a field's value is a valid primary key.
    """

    def __init__(self, field):
        super(PrimaryKeyConstraint, self).__init__(field)


class ForeignKeyConstraint(Constraint):
    """
    Constraint that checks whether a field's value is a valid foreign key.
    """

    def __enter__(self, field):
        super(ForeignKeyConstraint, self).__init__(field)


# unique constraint should have more than one field
class UniqueConstraint(Constraint):
    """
    Constraint that checks whether a field's value is unique.
    """

    def __init__(self, field):
        super(UniqueConstraint, self).__init__(field)


class CustomConstraint(Constraint):
    """
    Constraint that checks whether a field's value satisfies a custom condition.

    Attributes:
        validator (callable): A function that validates the field's value.
    """

    def __init__(self, field, validator=lambda x: True):
        super(CustomConstraint, self).__init__(field, validator)
