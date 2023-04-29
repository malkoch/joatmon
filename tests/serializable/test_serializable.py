import json

import pytest


def test_serializable_items():
    from joatmon.serializable import Serializable
    assert list(Serializable(a=12, b=13).items()) == [('a', 12), ('b', 13)]

    assert repr(Serializable(a=12, b=13)) == str(Serializable(a=12, b=13))
    assert Serializable(a=12, b=13)['a'] == 12
    assert Serializable(a=12, b=13).dict == {'a': 12, 'b': 13}
    assert Serializable.from_dict({'a': 12, 'b': 13})
    assert Serializable.from_dict({'a': 12, 'b': 13}).json == json.dumps({'a': 12, 'b': 13})
    assert Serializable.from_dict({'a': 12, 'b': 13}).pretty_json == json.dumps({'a': 12, 'b': 13}, indent=4)
    Serializable.from_bytes(Serializable.from_dict({'a': 12, 'b': 13}).bytes)
    assert Serializable.from_dict({'a': 12, 'b': 13}).snake
    assert Serializable.from_dict({'a': 12, 'b': 13}).pascal


def test_serializable_keys():
    from joatmon.serializable import Serializable
    assert list(Serializable(a=12, b=13).keys()) == ['a', 'b']


def test_serializable_values():
    from joatmon.serializable import Serializable
    assert list(Serializable(a=12, b=13).values()) == [12, 13]


if __name__ == '__main__':
    pytest.main([__file__])
