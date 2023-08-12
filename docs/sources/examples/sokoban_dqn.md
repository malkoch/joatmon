

```python
import gym

from joatmon.ai.models import DQNModel
from joatmon.ai.processor import RLProcessor
from joatmon.ai.trainer import DQNTrainer
from joatmon.callback import (
    CallbackList,
    Loader,
    Renderer,
    TrainLogger,
    ValidationLogger
)
from joatmon.game.sokoban import (
    SokobanEnv
)
from joatmon.ai.memory import RingMemory
from joatmon.ai.policy import (
    EpsilonGreedyPolicy as EGreedy,
    GreedyQPolicy as GreedyQ
)


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

    model = DQNModel(in_features=3, out_features=4)

    experiment = '.'
    case = '.'
    run_name = 'test'
    run_path = 'saves/01.dqn/{}/{}/{}/'.format(experiment, case, run_name)

    environment = create_env()
    agent = DQNTrainer(
        environment=environment,
        memory=memory,
        processor=processor,
        model=model,
        callbacks=CallbackList(
            [
                TrainLogger(run_path=run_path, interval=10),
                Loader(model=model, run_path=run_path, interval=10),
                Renderer(environment=environment)
            ]
        ),
        train_policy=EGreedy(min_value=0.1),
        test_policy=GreedyQ()
    )
    agent.train(max_episode=2400, warmup=1200000, max_action=200)

    environment = create_env()
    agent = DQNTrainer(
        environment=environment,
        memory=memory,
        processor=processor,
        model=model,
        callbacks=CallbackList(
            [
                ValidationLogger(run_path=run_path, interval=1),
                Renderer(environment=environment),
            ]
        ),
        train_policy=EGreedy(min_value=0.1),
        test_policy=GreedyQ()
    )
    agent.evaluate(max_action=200)


if __name__ == '__main__':
    run()
```
