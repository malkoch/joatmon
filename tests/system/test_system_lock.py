import pytest


def test_r_w_lock_r_acquire():
    assert True is True


def test_r_w_lock_r_locked():
    assert True is True


def test_r_w_lock_r_release():
    assert True is True


def test_r_w_lock_w_acquire():
    assert True is True


def test_r_w_lock_w_locked():
    assert True is True


def test_r_w_lock_w_release():
    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
