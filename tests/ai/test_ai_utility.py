import pytest


def test_display():
    assert True is True


def test_easy_range():
    from joatmon.ai.utility import easy_range

    for range_idx, easy_range_idx in zip(range(10), easy_range(end=10)):
        assert range_idx == easy_range_idx


def test_load():
    import torch
    from joatmon.ai.utility import (
        load,
        save
    )

    save(torch.nn.Linear(1, 2), 'weights/test')
    load(torch.nn.Linear(1, 2), 'weights/test')

    assert True is True


def test_normalize():
    import numpy as np
    from joatmon.ai.utility import normalize

    assert normalize(np.asarray([1, 2]), 0, 255).tolist() == np.asarray([0, 255]).astype('float32').tolist()


def test_range_tensor():
    from joatmon.ai.utility import range_tensor

    assert range_tensor(2).cpu().detach().numpy().tolist() == [0, 1]


def test_save():
    import torch
    from joatmon.ai.utility import save

    save(torch.nn.Linear(1, 2), 'weights/test')

    assert True is True


def test_to_numpy():
    from joatmon.ai.utility import (
        to_numpy,
        range_tensor
    )

    assert to_numpy(range_tensor(2)).tolist() == [0, 1]


def test_to_tensor():
    from joatmon.ai.utility import (
        to_numpy,
        range_tensor,
        to_tensor
    )

    assert to_tensor(to_numpy(range_tensor(2))).cpu().detach().numpy().tolist() == [0, 1]


if __name__ == '__main__':
    pytest.main([__file__])
