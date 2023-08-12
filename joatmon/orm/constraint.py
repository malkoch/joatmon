from joatmon.serializable import Serializable


class ValidationException(Exception):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """


class Constraint(Serializable):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, field, validator=None):
        super(Constraint, self).__init__()

        self.field = field
        self.validator = validator

    @staticmethod
    def create(constraint_type, **kwargs):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
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
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, field):
        super(PrimaryKeyConstraint, self).__init__(field)


class ForeignKeyConstrain(Constraint):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __enter__(self, field):
        super(ForeignKeyConstrain, self).__init__(field)


# unique constraint should have more than one field
class UniqueConstraint(Constraint):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, field):
        super(UniqueConstraint, self).__init__(field)


class CustomConstraint(Constraint):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, field, validator=lambda x: True):
        super(CustomConstraint, self).__init__(field, validator)
