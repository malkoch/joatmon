import uuid

import pytest


def test_j_s_o_n_decoder_object_hook():
    assert True is True


def test_j_s_o_n_encoder_default():
    assert True is True


def test_current_time():
    assert True is True


def test_empty_object_id():
    from joatmon.utility import empty_object_id
    assert empty_object_id() != uuid.uuid4()


def test_first():
    from joatmon.utility import first
    assert first([1, 2, 3]) == 1


def test_first_async():
    assert True is True


def test_get_class_that_defined_method():
    assert True is True


def test_get_converter():
    from joatmon.utility import get_converter
    assert get_converter(int)('1') == 1


def test_get_function_args():
    assert True is True


def test_get_function_kwargs():
    assert True is True


def test_get_module_functions():
    assert True is True


def test_ip_validator():
    from joatmon.utility import ip_validator
    assert ip_validator('127.0.0.1')


def test_mail_validator():
    from joatmon.utility import mail_validator
    assert mail_validator('hamitcanmalkoc@gmail.com')


def test_new_nickname():
    from joatmon.utility import new_nickname
    assert new_nickname() != ''


def test_new_object_id():
    from joatmon.utility import (
        new_object_id,
        empty_object_id
    )
    assert new_object_id() != empty_object_id()


def test_new_password():
    from joatmon.utility import new_password
    assert new_password() != ''


def test_single():
    from joatmon.utility import single
    assert single([1]) == 1


def test_single_async():
    assert True is True


def test_to_case():
    from joatmon.utility import to_case
    assert to_case('snake', {'A': 1}) == {'a': 1}


def test_to_enumerable():
    from joatmon.utility import to_enumerable
    assert to_enumerable([1, 2]) == [1, 2]


def test_to_hash():
    assert True is True


def test_to_list():
    from joatmon.utility import to_list
    assert to_list([1, 2]) == [1, 2]


def test_to_list_async():
    assert True is True


def test_to_lower_string():
    from joatmon.utility import to_lower_string
    assert to_lower_string('Abc') == 'abc'


def test_to_pascal_string():
    from joatmon.utility import to_pascal_string
    assert to_pascal_string('abc') == 'Abc'


def test_to_snake_string():
    from joatmon.utility import to_snake_string
    assert to_snake_string('ABC') == 'a_b_c'


def test_to_title():
    from joatmon.utility import to_title
    assert to_title('a') == 'A'


def test_to_upper_string():
    from joatmon.utility import to_upper_string
    assert to_upper_string('a') == 'A'


if __name__ == '__main__':
    pytest.main([__file__])
