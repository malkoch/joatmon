import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

from joatmon.ai.core import CoreModel
from joatmon.ai.utility import (
    load,
    save,
    to_numpy,
    to_tensor
)

__all__ = ['DDPGModel']


class DDPGModel(CoreModel):
    """
    Deep Deterministic Policy Gradient

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

    def __init__(self, lr=1e-3, tau=1e-4, gamma=0.99, actor=None, critic=None):
        super(DDPGModel, self).__init__()

        self.actor_local = actor
        self.actor_target = copy.deepcopy(actor)
        self.hardupdate('actor')

        self.critic_local = critic
        self.critic_target = copy.deepcopy(critic)
        self.hardupdate('critic')

        self.lr = lr
        self.tau = tau
        self.gamma = gamma

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr)
        self.critic_loss = nn.SmoothL1Loss()

    def load(self, path=''):
        load(self.actor_local, os.path.join(path, 'actor_local'))
        load(self.actor_target, os.path.join(path, 'actor_target'))
        load(self.critic_local, os.path.join(path, 'critic_local'))
        load(self.critic_target, os.path.join(path, 'critic_target'))

    def save(self, path=''):
        save(self.actor_local, os.path.join(path, 'actor_local'))
        save(self.actor_target, os.path.join(path, 'actor_target'))
        save(self.critic_local, os.path.join(path, 'critic_local'))
        save(self.critic_target, os.path.join(path, 'critic_target'))

    def initialize(self, w_init=None, b_init=None):
        for module in self.actor_local.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.actor_target.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.critic_local.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.critic_target.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)

    def softupdate(self, network: str):
        if network == 'actor':
            for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.tau) + param * self.tau)
        elif network == 'critic':
            for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                target_param.detach_()
                target_param.copy_(target_param * (1.0 - self.tau) + param * self.tau)
        else:
            raise ValueError

    def hardupdate(self, network: str):
        if network == 'actor':
            for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param.detach_()
                target_param.copy_(param)
        elif network == 'critic':
            for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                target_param.detach_()
                target_param.copy_(param)
        else:
            raise ValueError

    def predict(self, state=None):
        self.actor_local.eval()
        return to_numpy(self.actor_local(to_tensor(state))).flatten()

    def train(self, batch=None, update_target=True):
        self.actor_local.train()

        states, actions, rewards, next_states, terminals = batch

        states = to_tensor(states)
        actions = to_tensor(actions)
        rewards = to_tensor(rewards).unsqueeze(-1)
        next_states = to_tensor(next_states)
        terminals = to_tensor(terminals).unsqueeze(-1)

        a_next = self.actor_target(next_states)
        q_next = self.critic_target(next_states, a_next)

        q_next = self.gamma * q_next * (1 - terminals)
        q_next.add_(rewards)
        q_next = q_next.detach()

        q = self.critic_local(states, actions)

        critic_loss = self.critic_loss(q, q_next)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        a = self.actor_local(states)

        actor_loss = -self.critic_local(states, a).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_target:
            self.softupdate('actor')
            self.softupdate('critic')

        return [actor_loss.item()] + [critic_loss.item()]

    def evaluate(self):
        pass
