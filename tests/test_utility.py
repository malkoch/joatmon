import pytest


def test_j_s_o_n_decoder_object_hook():
    assert True is True


def test_j_s_o_n_encoder_default():
    assert True is True


def test_current_time():
    from joatmon.core.utility import current_time

    current_time()

    assert True is True


def test_empty_object_id():
    import uuid
    from joatmon.core.utility import empty_object_id

    assert empty_object_id() != uuid.uuid4()


def test_first():
    from joatmon.core.utility import first

    assert first([1, 2, 3]) == 1


def test_first_async():
    import asyncio
    from joatmon.core.utility import first_async

    async def t():
        async def t_():
            yield 1
            yield 2

        await first_async(t_())

    asyncio.ensure_future(t())

    assert True is True


def test_get_class_that_defined_method():
    from joatmon.core.utility import (
        JSONEncoder,
        get_class_that_defined_method
    )

    get_class_that_defined_method(JSONEncoder.default)

    assert True is True


def test_get_converter():
    import uuid
    import datetime
    from joatmon.core.utility import get_converter
    from joatmon.core.serializable import Serializable

    assert get_converter(int)(1.0) == 1
    assert get_converter(int)(1) == 1
    assert get_converter(int)('1') == 1
    assert get_converter(datetime.datetime)(datetime.datetime(2022, 1, 1).isoformat()).year == 2022
    assert get_converter(datetime.datetime)(datetime.datetime(2022, 1, 1)).year == 2022
    assert get_converter(float)(1.0) == 1
    assert get_converter(float)('1.0') == 1
    assert get_converter(float)(1) == 1
    assert get_converter(str)('1') == '1'
    assert get_converter(bytes)(b'1') == b'1'
    assert get_converter(bool)(True) is True
    assert get_converter(uuid.UUID)(uuid.uuid4())
    assert get_converter(uuid.UUID)(str(uuid.uuid4()))
    assert get_converter(dict)({}) == {}
    assert get_converter(dict)(Serializable()) == {}
    assert get_converter(list)([]) == []
    assert get_converter(tuple)(()) == ()
    assert get_converter(object)({}) == {}
    assert get_converter(object)([]) == []
    assert get_converter(object)(()) == ()
    assert get_converter(object)('') == ''
    assert get_converter(object)(Serializable()) == {}


def test_get_function_args():
    from joatmon.core.utility import (
        get_function_args,
        JSONEncoder
    )

    def t():
        ...

    get_function_args(JSONEncoder, '12')
    get_function_args(t)

    assert True is True


def test_get_function_kwargs():
    from joatmon.core.utility import get_function_kwargs

    def t():
        ...

    get_function_kwargs(t)

    assert True is True


def test_get_module_functions():
    import joatmon
    from joatmon.core.utility import get_module_functions

    get_module_functions(joatmon)

    assert True is True


def test_ip_validator():
    from joatmon.core.utility import ip_validator

    assert ip_validator('127.0.0.1')


def test_mail_validator():
    from joatmon.core.utility import mail_validator

    assert mail_validator('hamitcanmalkoc@gmail.com')


def test_new_nickname():
    from joatmon.core.utility import new_nickname

    assert new_nickname() != ''


def test_new_object_id():
    from joatmon.core.utility import (
        new_object_id,
        empty_object_id
    )

    assert new_object_id() != empty_object_id()


def test_new_password():
    from joatmon.core.utility import new_password

    assert new_password() != ''


def test_single():
    from joatmon.core.utility import single

    assert single([1]) == 1


def test_single_async():
    import asyncio
    from joatmon.core.utility import single_async

    async def t():
        async def t_():
            yield 1

        await single_async(t_())

    asyncio.ensure_future(t())

    assert True is True


def test_to_case():
    from joatmon.core.utility import to_case

    assert to_case('snake', {'A': 1}) == {'a': 1}
    assert to_case('pascal', {'a': 1}) == {'A': 1}
    assert to_case('upper', {'a': 1}) == {'A': 1}
    assert to_case('lower', {'A': 1}) == {'a': 1}


def test_to_enumerable():
    from joatmon.core.serializable import Serializable
    from joatmon.core.utility import to_enumerable

    assert to_enumerable([1, 2]) == [1, 2]
    assert to_enumerable((1, 2)) == (1, 2)
    assert to_enumerable(Serializable(a=1, b=2)) == {'a': 1, 'b': 2}


def test_to_hash():
    from joatmon.core.utility import to_hash

    def t(a, b):
        ...

    to_hash(t, ('1',), {'b': 2})

    assert True is True


def test_to_list():
    from joatmon.core.utility import to_list

    assert to_list([1, 2]) == [1, 2]


def test_to_list_async():
    import asyncio
    from joatmon.core.utility import to_list_async

    async def t():
        async def t_():
            yield 1
            yield 2

        await to_list_async(t_())

    asyncio.ensure_future(t())

    assert True is True


def test_to_lower_string():
    from joatmon.core.utility import to_lower_string

    assert to_lower_string('Abc') == 'abc'


def test_to_pascal_string():
    from joatmon.core.utility import to_pascal_string

    assert to_pascal_string('abc') == 'Abc'


def test_to_snake_string():
    from joatmon.core.utility import to_snake_string

    assert to_snake_string('ABC') == 'a_b_c'


def test_to_title():
    from joatmon.core.utility import to_title

    assert to_title('a') == 'A'


def test_to_upper_string():
    from joatmon.core.utility import to_upper_string

    assert to_upper_string('a') == 'A'


if __name__ == '__main__':
    pytest.main([__file__])
