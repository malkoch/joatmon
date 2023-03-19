class meta(type):
    def __new__(
            upperattr_metaclass, future_class_name,
            future_class_parents, future_class_attrs
    ):
        uppercase_attrs = {
            attr if attr.startswith("__") else attr.upper(): v
            for attr, v in future_class_attrs.items()
        }
        return type(future_class_name, future_class_parents, uppercase_attrs)

    def __instancecheck__(self, instance):
        print(f'checking instance of {instance}')


class Term(metaclass=meta):

    def __init__(self):
        ...

    def __instancecheck__(self, instance):
        print(f'checking instance of {instance}')


print(Term() is True)
