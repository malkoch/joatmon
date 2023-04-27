import pytest


def test_r_l_processor_process_batch():
    import numpy as np
    from joatmon.ai.processor import RLProcessor

    processor = RLProcessor()
    processor.process_batch(
        [(
            np.random.random((1, 3, 84, 84)),
            np.random.random((1, 1)),
            np.random.random((1, 1)),
            np.random.random((1, 3, 84, 84)),
            np.random.random((1, 1))
        )]
    )

    assert True is True


def test_r_l_processor_process_state():
    import numpy as np
    from joatmon.ai.processor import RLProcessor

    processor = RLProcessor()
    processor.process_state(np.random.randint(0, 255, (84, 84, 3)))

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
