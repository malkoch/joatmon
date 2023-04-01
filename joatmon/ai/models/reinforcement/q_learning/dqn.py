import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from joatmon.ai.models.core import CoreModel
from joatmon.ai.utility import (
    load,
    range_tensor,
    save,
    to_numpy,
    to_tensor
)

__all__ = ['DQNModel']


class DQNModel(CoreModel):
    """
    Deep Q Network

    # Arguments
        models (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, lr=1e-3, tau=1e-4, gamma=0.99, network=None):
        super().__init__()

        self.local = network
        self.target = copy.deepcopy(network)
        self.hardupdate()

        self.lr = lr
        self.tau = tau
        self.gamma = gamma

        self.optimizer = optim.Adam(self.local.parameters(), lr=self.lr)
        self.loss = nn.SmoothL1Loss()

    def load(self, path=''):
        load(self.local, os.path.join(path, 'local'))
        load(self.target, os.path.join(path, 'target'))

    def save(self, path=''):
        save(self.local, os.path.join(path, 'local'))
        save(self.target, os.path.join(path, 'target'))

    def initialize(self, w_init=None, b_init=None):
        for module in self.target.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)
        for module in self.local.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)

    def softupdate(self):
        for target_param, param in zip(self.target.parameters(), self.local.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.tau) + param * self.tau)

    def hardupdate(self):
        for target_param, param in zip(self.target.parameters(), self.local.parameters()):
            target_param.detach_()
            target_param.copy_(param)

    def predict(self, state=None):
        self.local.eval()
        return np.argmax(to_numpy(self.local(to_tensor(state))).flatten())

    def train(self, batch=None, update_target=False):
        self.local.train()

        states, actions, rewards, next_states, terminals = batch

        states = to_tensor(states)
        actions = to_tensor(actions)
        rewards = to_tensor(rewards)
        next_states = to_tensor(next_states)
        terminals = to_tensor(terminals)

        batch_indices = range_tensor(states.size(0))

        q_next = self.target(next_states).detach()
        q_next = q_next.max(1)[0]
        q_next = self.gamma * q_next * (1 - terminals)
        q_next.add_(rewards)

        q = self.local(states)
        q = q[batch_indices, actions.long()]
        loss = self.loss(q, q_next)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.local.parameters(), 1)
        self.optimizer.step()

        if update_target:
            self.softupdate()

        return loss.item()

    def evaluate(self):
        pass
