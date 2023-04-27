import pytest


def test_d_q_n_model_evaluate():
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.evaluate()

    assert True is True


def test_d_q_n_model_hardupdate():
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.hardupdate()

    assert True is True


def test_d_q_n_model_initialize():
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.initialize()

    assert True is True


def test_d_q_n_model_load():
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.save('weights/dqn')
    model.load('weights/dqn')

    assert True is True


def test_d_q_n_model_predict():
    import numpy as np
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.predict(np.random.random((1, 3, 84, 84)))

    assert True is True


def test_d_q_n_model_save():
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.save('weights/dqn')

    assert True is True


def test_d_q_n_model_softupdate():
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.softupdate()

    assert True is True


def test_d_q_n_model_train():
    import numpy as np
    from joatmon.ai.models.reinforcement.q_learning.dqn import DQNModel

    model = DQNModel(in_features=3, out_features=1)
    model.train(
        [
            np.random.random((32, 3, 84, 84)),
            np.random.random((32, 1)),
            np.random.random((32, 1)),
            np.random.random((32, 3, 84, 84)),
            np.random.random((32, 1))
        ], True
    )

    assert True is True


if __name__ == '__main__':
    pytest.main([__file__])
