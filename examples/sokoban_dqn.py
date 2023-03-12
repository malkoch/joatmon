import copy
import gc
import math

import gym
import numpy as np
import pygame
import torch.nn as nn
from numpy import linalg
from PIL import Image

from joatmon.ai.callback import (
    CallbackList,
    Loader,
    Renderer,
    TrainLogger,
    ValidationLogger
)
from joatmon.ai.memory import RingMemory
from joatmon.ai.models import DQNModel
from joatmon.ai.policy import (
    EpsilonGreedyPolicy as EGreedy,
    GreedyQPolicy as GreedyQ
)
from joatmon.ai.utility import easy_range
from joatmon.game.sokoban import (
    draw_circle,
    draw_rectangle,
    draw_sprite,
    SokobanEnv
)


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


class DQN(nn.Module):
    def __init__(self, in_features, out_features):
        super(DQN, self).__init__()

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
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def forward(self, x):
        return self.predictor(self.extractor(x))


class DQNTrainer:
    def __init__(self, environment, memory, processor, model, callbacks, test_policy, train_policy, her=False, action_num=4):
        super(DQNTrainer, self).__init__()

        self.environment = environment
        self.memory = memory
        self.processor = processor
        self.model = model
        self.callbacks = callbacks
        self.test_policy = test_policy
        self.train_policy = train_policy
        self.her = her
        self.action_number = action_num

    def final_state(self):
        screen = pygame.Surface(self.environment.world_metrics)
        screen.fill(self.environment.ground_color)

        for obstacle in self.environment.obstacles:
            if self.environment.obstacle_sprite is not None:
                draw_sprite(
                    screen,
                    image=self.environment.obstacle_sprite,
                    position=obstacle.body.position,
                    half_size=self.environment.obj_size * 0.5
                )
            else:
                draw_rectangle(
                    screen,
                    color=self.environment.obstacle_color,
                    position=obstacle.body.position,
                    half_size=self.environment.obj_size * 0.5
                )

        player = self.environment.players[0]
        for goal in self.environment.goals:
            if self.environment.goal_sprite is not None:
                draw_sprite(
                    screen,
                    image=self.environment.goal_sprite,
                    position=goal.body.position,
                    half_size=self.environment.obj_size * 0.5
                )
            else:
                draw_rectangle(
                    screen,
                    color=self.environment.goal_color,
                    position=goal.body.position,
                    half_size=self.environment.obj_size * 0.5
                )

            if self.environment.block_sprite is not None:
                draw_sprite(
                    screen,
                    image=self.environment.block_sprite,
                    position=player.body.position - (player.body.position - goal.body.position) * 0.95,
                    half_size=self.environment.obj_size * 0.5
                )
            else:
                draw_rectangle(
                    screen,
                    color=self.environment.block_color,
                    position=player.body.position - (player.body.position - goal.body.position) * 0.95,
                    half_size=self.environment.obj_size * 0.5
                )

        player = self.environment.players[0]
        block = self.environment.blocks[0]
        if self.environment.player_sprite is not None:
            draw_sprite(
                screen, image=self.environment.player_sprite, position=player.body.position - (player.body.position - block.body.position) * 0.9,
                half_size=self.environment.obj_size * 0.5
                )
        else:
            draw_circle(
                screen, color=self.environment.player_color, position=player.body.position - (player.body.position - block.body.position) * 0.9,
                radius=self.environment.obj_size * 0.5
                )

        image = pygame.surfarray.array3d(screen)
        image = np.swapaxes(image, 0, 1)
        return self.processor.process_state(image)

    def get_step(self, action, mode='discrete', action_number=4):
        if mode == 'discrete':
            degree_inc = 360.0 / action_number
            degree = action * degree_inc
            radian = math.radians(degree)
            step = [math.cos(radian), math.sin(radian)]
        else:
            norm = linalg.norm(action)
            if norm >= 1.0:
                action /= norm
            step = action
        return step

    def get_action(self, state, goal_state, policy):
        if self.her:
            s = np.concatenate((state, goal_state), axis=2)
        else:
            s = state

        if policy.use_network():
            action = self.model.predict(s)
        else:
            action = np.random.randint(0, self.action_number)

        return action

    def train(self, batch_size=32, max_action=200, max_episode=12000, warmup=120000):
        total_steps = 0
        self.callbacks.on_agent_begin(
            **{
                'agent_headers': ['episode_number', 'action_number', 'episode_reward'],
                'network_headers': ['loss']
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
                action = self.get_action(state, goal_state, self.train_policy)
                self.callbacks.on_action_begin(
                    **{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'state': state,
                        'action': action
                    }
                    )

                step = self.get_step(action, 'discrete', self.action_number)
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
                self.memory.remember((processed_state, action, clipped_reward, processed_next_state, terminal))

                if total_steps > warmup:
                    self.train_policy.decay()
                    if total_steps % batch_size == 0:
                        self.callbacks.on_replay_begin()
                        mini_batch = self.memory.sample()
                        batch = self.processor.process_batch(mini_batch)
                        loss = self.model.train(batch)
                        self.callbacks.on_replay_end(
                            **{
                                'loss': loss
                            }
                            )

                episode_reward += reward
                state = copy.deepcopy(next_state)
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
                action = self.get_action(state, goal_state, self.test_policy)
                self.callbacks.on_action_begin(
                    **{
                        'episode_number': episode_number,
                        'action_number': action_number,
                        'state': state,
                        'action': action
                    }
                    )

                step = self.get_step(action, 'discrete', self.action_number)
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
                state = copy.deepcopy(next_state)
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

    model = DQNModel(network=DQN(in_features=3, out_features=4))

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
