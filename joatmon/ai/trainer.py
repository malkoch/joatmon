import copy
import gc
import math
from copy import deepcopy

import numpy as np
from numpy import linalg

from joatmon.ai.utility import easy_range


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

    def goal(self):
        goal = self.environment.goal()
        if goal:
            return self.processor.process_state(goal)

    def get_step(self, action, mode='q_learning', action_number=4):
        if mode == 'q_learning':
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

            goal_state = self.goal()

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

                step = self.get_step(action, 'q_learning', self.action_number)
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

            goal_state = self.goal()

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

                step = self.get_step(action, 'q_learning', self.action_number)
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

    def goal(self):
        goal = self.environment.goal()
        if goal:
            return self.processor.process_state(goal)

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

            goal_state = self.goal()

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
                    step = self.environment.get_step(action, 'policy_optimization')
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

            goal_state = self.goal()

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
                    step = self.environment.get_step(action, 'policy_optimization')
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
