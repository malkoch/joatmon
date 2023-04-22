import pytest


def test_binary_search():
    from joatmon.algoritm.search.binary import binary_search

    assert binary_search([0, 1, 2, 3], 1, lambda x: x) == 1


def test_binary_search_helper():
    assert True is True


def test_binary_search_r():
    from joatmon.algoritm.search.binary import binary_search_r

    assert binary_search_r([0, 1, 2, 3], 1, lambda x: x) == 1


if __name__ == '__main__':
    pytest.main([__file__])
