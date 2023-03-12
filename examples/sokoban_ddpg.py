import gc
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from joatmon.ai.callback import (
    CallbackList,
    Loader,
    Renderer,
    TrainLogger,
    ValidationLogger
)
from joatmon.ai.memory import RingMemory
from joatmon.ai.models import DDPGModel
from joatmon.ai.random import OrnsteinUhlenbeck as RandomProcess
from joatmon.ai.utility import easy_range
from joatmon.game import SokobanEnv


class RLProcessor:
    def __init__(self):
        super(RLProcessor, self).__init__()

    @staticmethod
    def process_batch(batch):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for state, action, reward, next_state, terminal in batch:
            states.extend(state)
            actions.append(action)
            rewards.append(reward)
            next_states.extend(next_state)
            terminals.append(terminal)
        states = np.asarray(states).astype('float32')
        actions = np.asarray(actions).astype('float32')
        rewards = np.asarray(rewards).astype('float32')
        next_states = np.asarray(next_states).astype('float32')
        terminals = np.asarray(terminals).astype('float32')
        return states, actions, rewards, next_states, terminals

    @staticmethod
    def process_state(state):
        state = Image.fromarray(state)
        state = state.resize((84, 84))
        state = np.array(state)
        state = np.expand_dims(state, 0)
        state = np.transpose(state, (0, 3, 1, 2))
        state = state.astype('uint8')
        return state


class DDPGActor(nn.Module):
    def __init__(self, in_features, out_features):
        super(DDPGActor, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.predictor = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=200),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=200),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=200, out_features=out_features),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.extractor(x)
        return self.predictor(x.view(x.size(0), -1))


class DDPGCritic(nn.Module):
    def __init__(self, in_features, out_features):
        super(DDPGCritic, self).__init__()

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(in_features=7 * 7 * 64, out_features=200)
        self.linear2 = nn.Linear(in_features=200 + out_features, out_features=200)
        self.linear3 = nn.Linear(in_features=200, out_features=1)

        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(200)

    def forward(self, x, y):
        x = self.extractor(x)

        x = self.relu(self.bn1(self.linear1(x.view(x.size(0), -1))))
        x = torch.cat([x, y], dim=1)
        x = self.relu(self.bn2(self.linear2(x)))
        return self.linear3(x)


class DDPGTrainer:
    def __init__(self, environment, random_process, processor, memory, model, callbacks, her=False):
        super(DDPGTrainer, self).__init__()

        self.environment = environment
        self.memory = memory
        self.processor = processor
        self.model = model
        self.callbacks = callbacks
        self.random_process = random_process
        self.her = her

    def final_state(self):
        if hasattr(self.environment, 'final_state'):
            return self.processor.process_state(self.environment.final_state())
        return None

    def get_action(self, state, goal_state):
        if self.her:
            action = self.model.predict(np.concatenate((state, goal_state), axis=2))
        else:
            action = self.model.predict(state)
        action += self.random_process.sample()

        return action

    def train(self, batch_size=32, max_action=50, max_episode=120, warmup=0, replay_interval=4, update_interval=1, test_interval=1000):
        total_steps = 0
        self.callbacks.on_agent_begin(
            **{
                'agent_headers': ['episode_number', 'action_number', 'episode_reward'],
                'network_headers': ['actor_loss', 'critic_loss', 'critic_extra_loss']
            }
            )
        for episode_number in easy_range(1, max_episode):
            episode_reward = 0
            state = self.environment.reset()
            state = self.processor.process_state(state)
            self.callbacks.on_episode_begin(
                **{
                    'episode_number': episode_number,
                    'state': state
                }
                )

            goal_state = self.final_state()

            for action_number in easy_range(1, max_action):
                action = self.get_action(state, goal_state)
                self.callbacks.on_action_begin(
                    **{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'state': state,
                        'action': action
                    }
                    )

                if hasattr(self.environment, 'get_step'):
                    step = self.environment.get_step(action, 'continuous')
                else:
                    step = action
                next_state, reward, terminal, _ = self.environment.step(step)
                next_state = self.processor.process_state(next_state)
                if action_number >= max_action:
                    terminal = True

                self.callbacks.on_action_end(
                    **{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'terminal': terminal,
                        'next_state': next_state
                    }
                    )

                processed_state = np.concatenate((state, goal_state), axis=2) if self.her else state
                clipped_reward = np.clip(reward - 0.25, -1, 1)
                processed_next_state = np.concatenate((next_state, goal_state), axis=2) if self.her else next_state
                self.memory.remember(processed_state, action, clipped_reward, processed_next_state, terminal)

                if total_steps > warmup:
                    self.random_process.decay()
                    if total_steps % replay_interval == 0:
                        self.callbacks.on_replay_begin()
                        mini_batch = self.memory.sample()
                        batch = self.processor.process_batch(mini_batch)
                        loss = self.model.train(batch, ((total_steps - warmup) // replay_interval) % update_interval == 0)
                        self.callbacks.on_replay_end(
                            **{
                                'loss': loss
                            }
                            )

                episode_reward += reward
                state = deepcopy(next_state)
                total_steps += 1

                if terminal:
                    self.callbacks.on_episode_end(
                        **{
                            'episode_number': episode_number,
                            'action_number': action_number,
                            'episode_reward': episode_reward
                        }
                        )
                    gc.collect()
                    break

        self.environment.close()
        self.callbacks.on_agent_end(
            **{
                'total_steps': total_steps
            }
            )

    def evaluate(self, max_action=50, max_episode=12):
        total_steps = 0
        self.callbacks.on_agent_begin()
        for episode_number in easy_range(1, max_episode):
            episode_reward = 0
            state = self.environment.reset()
            state = self.processor.process_state(state)
            self.callbacks.on_episode_begin(
                **{
                    'episode_number': episode_number,
                    'state': state
                }
                )

            goal_state = self.final_state()

            for action_number in easy_range(1, max_action):
                action = self.get_action(state, goal_state)
                self.callbacks.on_action_begin(
                    **{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'state': state,
                        'action': action
                    }
                    )

                if hasattr(self.environment, 'get_step'):
                    step = self.environment.get_step(action, 'continuous')
                else:
                    step = action
                next_state, reward, terminal, _ = self.environment.step(step)
                next_state = self.processor.process_state(next_state)
                if action_number >= max_action:
                    terminal = True

                self.callbacks.on_action_end(
                    **{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'terminal': terminal,
                        'next_state': next_state
                    }
                    )

                episode_reward += reward
                state = deepcopy(next_state)
                total_steps += 1

                if terminal:
                    self.callbacks.on_episode_end(
                        **{
                            'episode_number': episode_number,
                            'action_number': action_number,
                            'episode_reward': episode_reward
                        }
                        )

                    gc.collect()
                    break

        self.environment.close()
        self.callbacks.on_agent_end(
            **{
                'total_steps': total_steps
            }
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

    model = DDPGModel(tau=1e-3, actor=DDPGActor(in_features=3, out_features=2), critic=DDPGCritic(in_features=3, out_features=2))

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
    agent = DDPGTrainer(
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
    agent = DDPGTrainer(
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
