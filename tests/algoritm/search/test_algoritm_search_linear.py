import pytest


def test_linear_search():
    from joatmon.algoritm.search.linear import linear_search

    assert linear_search([0, 1, 2, 3], 1, lambda x: x) == 1


if __name__ == '__main__':
    pytest.main([__file__])
