

```python
import gym

from joatmon.ai.models.reinforcement.hybrid.td3 import TD3Model
from joatmon.ai.processor import RLProcessor
from joatmon.ai.trainer import TD3Trainer
from joatmon.callback import (
    CallbackList,
    Loader,
    Renderer,
    TrainLogger,
    ValidationLogger
)
from joatmon.game import SokobanEnv
from joatmon.ai.memory import RingMemory
from joatmon.ai.random import OrnsteinUhlenbeck as RandomProcess


def create_env():
    try:
        environment = gym.make(
            'Sokoban-Medium-v0', **{
                'xmls': 'game/assets/sokoban/xmls/',
                'sprites': 'game/assets/sokoban/sprites/'
            }
        )
    except Exception as ex:
        print(str(ex))
        environment = SokobanEnv(
            **{
                'xml': 'medium.xml',
                'xmls': 'game/assets/sokoban/xmls/',
                'sprites': 'game/assets/sokoban/sprites/'
            }
        )
    return environment


def run():
    memory = RingMemory()
    processor = RLProcessor()

    model = TD3Model(tau=1e-3, in_features=3, out_features=2)

    experiment = '.'
    case = '.'
    run_name = 'test'
    run_path = 'saves/02.ddpg/{}/{}/{}/'.format(experiment, case, run_name)

    random_process = RandomProcess(decay_steps=1200000)
    environment = create_env()
    callbacks = CallbackList(
        [
            TrainLogger(run_path=run_path, interval=100),
            Loader(model=model, run_path=run_path, interval=1000),
            Renderer(environment=environment)
        ]
    )
    agent = TD3Trainer(
        environment=environment,
        memory=memory,
        processor=processor,
        model=model,
        callbacks=callbacks,
        random_process=random_process
    )
    agent.train(max_action=200, max_episode=2, warmup=12, replay_interval=32)

    random_process = RandomProcess(sigma=0.0, sigma_min=0.0, decay_steps=1200000)
    environment = create_env()
    callbacks = CallbackList(
        [
            ValidationLogger(run_path=run_path, interval=1),
            Loader(model=model, run_path=run_path, interval=1000),
            Renderer(environment=environment),
        ]
    )
    agent = TD3Trainer(
        environment=environment,
        memory=memory,
        processor=processor,
        model=model,
        callbacks=callbacks,
        random_process=random_process
    )
    agent.evaluate(max_action=200)


if __name__ == '__main__':
    run()
```