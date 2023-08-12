import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from joatmon.ai.models.core import CoreModel
from joatmon.ai.network.reinforcement.hybrid.td3 import (
    TD3Actor,
    TD3Critic
)
from joatmon.ai.utility import (
    load,
    save,
    to_numpy,
    to_tensor
)

__all__ = ['TD3Model']


class TD3Model(CoreModel):
    """
    Twin Delayed Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, lr=1e-3, tau=1e-4, gamma=0.99, in_features=1, out_features=1):
        super(TD3Model, self).__init__()

        self.actor_local = TD3Actor(in_features, out_features)
        self.actor_target = copy.deepcopy(self.actor_local)
        self.hardupdate('actor')

        self.critic_local_1 = TD3Critic(in_features, out_features)
        self.critic_target_1 = copy.deepcopy(self.critic_local_1)
        self.hardupdate('critic_1')

        self.critic_local_2 = TD3Critic(in_features, out_features)
        self.critic_target_2 = copy.deepcopy(self.critic_local_2)
        self.hardupdate('critic_2')

        self.lr = lr
        self.tau = tau
        self.gamma = gamma

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr)

        self.critic_optimizer = optim.Adam(
            list(self.critic_local_1.parameters()) + list(self.critic_local_2.parameters()), lr=self.lr
        )
        self.critic_loss = nn.SmoothL1Loss()

    def load(self, path=''):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        load(self.actor_local, os.path.join(path, 'actor_local'))
        load(self.actor_target, os.path.join(path, 'actor_target'))
        load(self.critic_local_1, os.path.join(path, 'critic_local_1'))
        load(self.critic_target_1, os.path.join(path, 'critic_target_1'))
        load(self.critic_local_2, os.path.join(path, 'critic_local_2'))
        load(self.critic_target_2, os.path.join(path, 'critic_target_2'))

    def save(self, path=''):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        save(self.actor_local, os.path.join(path, 'actor_local'))
        save(self.actor_target, os.path.join(path, 'actor_target'))
        save(self.critic_local_1, os.path.join(path, 'critic_local_1'))
        save(self.critic_target_1, os.path.join(path, 'critic_target_1'))
        save(self.critic_local_2, os.path.join(path, 'critic_local_2'))
        save(self.critic_target_2, os.path.join(path, 'critic_target_2'))

    def initialize(self, w_init=None, b_init=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for module in self.actor_local.modules():
            if isinstance(
                    module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.actor_target.modules():
            if isinstance(
                    module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.critic_local_1.modules():
            if isinstance(
                    module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.critic_target_1.modules():
            if isinstance(
                    module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.critic_local_2.modules():
            if isinstance(
                    module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.critic_target_2.modules():
            if isinstance(
                    module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)

    def softupdate(self, network: str):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if network == 'actor':
            for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.tau) + param * self.tau)
        elif network == 'critic_1':
            for target_param, param in zip(self.critic_target_1.parameters(), self.critic_local_1.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.tau) + param * self.tau)
        elif network == 'critic_2':
            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.tau) + param * self.tau)
        else:
            raise ValueError

    def hardupdate(self, network: str):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if network == 'actor':
            for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param.detach_()
                target_param.copy_(param)
        elif network == 'critic_1':
            for target_param, param in zip(self.critic_target_1.parameters(), self.critic_local_1.parameters()):
                target_param.detach_()
                target_param.copy_(param)
        elif network == 'critic_2':
            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
                target_param.detach_()
                target_param.copy_(param)
        else:
            raise ValueError

    def predict(self, state=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.actor_local.eval()
        return to_numpy(self.actor_local(to_tensor(state))).flatten()

    def train(self, batch=None, update_target=True):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.actor_local.train()

        states, actions, rewards, next_states, terminals = batch

        states = to_tensor(states)
        actions = to_tensor(actions)
        rewards = to_tensor(rewards).unsqueeze(-1)
        next_states = to_tensor(next_states)
        terminals = to_tensor(terminals).unsqueeze(-1)

        a_next = self.actor_target(next_states)
        q_next_1 = self.critic_target_1(next_states, a_next)
        q_next_2 = self.critic_target_2(next_states, a_next)
        q_next = torch.min(q_next_1, q_next_2)

        q_next = self.gamma * q_next * (1 - terminals)
        q_next.add_(rewards)
        q_next = q_next.detach()

        q_1 = self.critic_local_1(states, actions)
        q_2 = self.critic_local_2(states, actions)

        critic_1_loss = F.mse_loss(q_1, q_next)
        critic_2_loss = F.mse_loss(q_2, q_next)

        critic_loss = critic_1_loss + critic_2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # might need to update with policy update frequency
        a = self.actor_local(states)

        actor_loss = -self.critic_local_1(states, a).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_target:
            self.softupdate('actor')
            self.softupdate('critic_1')
            self.softupdate('critic_2')

        return [actor_loss.item()] + [critic_loss.item()]

    def evaluate(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
