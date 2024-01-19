import copy
import os

import numpy as np
from joatmon.nn import init

from joatmon.ai.models.core import CoreModel
from joatmon.ai.network import DQN
from joatmon.ai.utility import (
    load,
    range_tensor,
    save,
    to_numpy,
    to_tensor
)

__all__ = ['DQNModel']

from joatmon.nn.layer.batchnorm import BatchNorm

from joatmon.nn.layer.conv import Conv
from joatmon.nn.layer.linear import Linear

from joatmon.nn.loss.huber import HuberLoss

from joatmon.nn.optimizer.adam import Adam


class DQNModel(CoreModel):
    """
    Deep Q Network Model.

    This class implements the DQN model, which is a type of reinforcement learning model.
    It inherits from the CoreModel class and overrides its abstract methods.

    # Arguments
        lr (float): The learning rate for the Adam optimizer.
        tau (float): The factor for soft update of target parameters.
        gamma (float): The discount factor.
        in_features (int): The number of input features.
        out_features (int): The number of output features.
    """

    def __init__(self, lr=1e-3, tau=1e-4, gamma=0.99, in_features=1, out_features=1):
        """
        Constructor for the DQNModel class.

        Initializes the local and target networks.
        Also initializes the optimizer and loss function for the networks.
        """
        super().__init__()

        self.local = DQN(in_features, out_features)
        self.target = copy.deepcopy(self.local)
        self.hardupdate()

        self.lr = lr
        self.tau = tau
        self.gamma = gamma

        self.optimizer = Adam(list(self.local.parameters()), lr=self.lr)
        self.loss = HuberLoss()

    def load(self, path=''):
        """
        Load the local and target networks from the specified path.

        # Arguments
            path (str): The path to the directory where the networks are stored.
        """
        load(self.local, os.path.join(path, 'local'))
        load(self.target, os.path.join(path, 'target'))

    def save(self, path=''):
        """
        Save the local and target networks to the specified path.

        # Arguments
            path (str): The path to the directory where the networks should be saved.
        """
        save(self.local, os.path.join(path, 'local'))
        save(self.target, os.path.join(path, 'target'))

    def initialize(self, w_init=None, b_init=None):
        """
        Initialize the weights and biases of the local and target networks.

        # Arguments
            w_init (callable, optional): The function to use for initializing the weights.
            b_init (callable, optional): The function to use for initializing the biases.
        """
        for module in self.target.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)
        for module in self.local.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)

    def softupdate(self):
        """
        Perform a soft update of the target network parameters.
        """
        for target_param, param in zip(self.target.parameters(), self.local.parameters()):
            target_param._data = target_param.data * (1.0 - self.tau) + param.data * self.tau

    def hardupdate(self):
        """
        Perform a hard update of the target network parameters.
        """
        for target_param, param in zip(self.target.parameters(), self.local.parameters()):
            target_param._data = param.data

    def predict(self, state=None):
        """
        Get the action for a given state.

        # Arguments
            state (array-like, optional): The current state.

        # Returns
            action (array-like): The predicted action for the given state.
        """
        self.local.eval()
        return np.argmax(to_numpy(self.local(to_tensor(state))).flatten())

    def train(self, batch=None, update_target=False):
        """
        Train the local network with a given batch.

        # Arguments
            batch (tuple of array-like, optional): The batch of experiences to train on.
            update_target (bool, optional): Whether to perform a soft update of the target network.

        # Returns
            loss (float): The loss of the local network.
        """
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
        # nn.utils.clip_grad_norm_(self.local.parameters(), 1)
        self.optimizer.step()

        if update_target:
            self.softupdate()

        return loss.item()

    def evaluate(self):
        """
        Evaluate the local and target networks.

        This method should be overridden in a subclass.
        """
