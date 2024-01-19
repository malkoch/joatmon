import copy
import os

from joatmon.ai.models.core import CoreModel
from joatmon.ai.network.td3 import (
    TD3Actor,
    TD3Critic
)
from joatmon.ai.utility import (
    load,
    save,
    to_numpy,
    to_tensor
)
from joatmon.nn import functional, init
from joatmon.nn.layer.batchnorm import BatchNorm
from joatmon.nn.layer.conv import Conv
from joatmon.nn.layer.linear import Linear
from joatmon.nn.loss.huber import HuberLoss
from joatmon.nn.optimizer.adam import Adam

__all__ = ['TD3Model']


class TD3Model(CoreModel):
    """
    Twin Delayed Deep Deterministic Policy Gradient Model.

    This class implements the TD3 model, which is a type of reinforcement learning model.
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
        Constructor for the TD3Model class.

        Initializes the actor and critic networks, both local and target.
        Also initializes the optimizers and loss function for the networks.
        """
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

        self.actor_optimizer = Adam(list(self.actor_local.parameters()), lr=self.lr)

        self.critic_optimizer = Adam(
            list(self.critic_local_1.parameters()) + list(self.critic_local_2.parameters()), lr=self.lr
        )
        self.critic_loss = HuberLoss()

    def load(self, path=''):
        """
        Load the actor and critic networks from the specified path.

        # Arguments
            path (str): The path to the directory where the networks are stored.
        """
        load(self.actor_local, os.path.join(path, 'actor_local'))
        load(self.actor_target, os.path.join(path, 'actor_target'))
        load(self.critic_local_1, os.path.join(path, 'critic_local_1'))
        load(self.critic_target_1, os.path.join(path, 'critic_target_1'))
        load(self.critic_local_2, os.path.join(path, 'critic_local_2'))
        load(self.critic_target_2, os.path.join(path, 'critic_target_2'))

    def save(self, path=''):
        """
        Save the actor and critic networks to the specified path.

        # Arguments
            path (str): The path to the directory where the networks should be saved.
        """
        save(self.actor_local, os.path.join(path, 'actor_local'))
        save(self.actor_target, os.path.join(path, 'actor_target'))
        save(self.critic_local_1, os.path.join(path, 'critic_local_1'))
        save(self.critic_target_1, os.path.join(path, 'critic_target_1'))
        save(self.critic_local_2, os.path.join(path, 'critic_local_2'))
        save(self.critic_target_2, os.path.join(path, 'critic_target_2'))

    def initialize(self, w_init=None, b_init=None):
        """
        Initialize the weights and biases of the actor and critic networks.

        # Arguments
            w_init (callable, optional): The function to use for initializing the weights.
            b_init (callable, optional): The function to use for initializing the biases.
        """
        for module in self.actor_local.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)
        for module in self.actor_target.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)
        for module in self.critic_local_1.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)
        for module in self.critic_target_1.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)
        for module in self.critic_local_2.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)
        for module in self.critic_target_2.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)

    def softupdate(self, network: str):
        """
        Perform a soft update of the target network parameters.

        # Arguments
            network (str): The name of the network to update ('actor' or 'critic_1' or 'critic_2').
        """
        if network == 'actor':
            for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param._data = target_param.data * (1.0 - self.tau) + param.data * self.tau
        elif network == 'critic_1':
            for target_param, param in zip(self.critic_target_1.parameters(), self.critic_local_1.parameters()):
                target_param._data = target_param.data * (1.0 - self.tau) + param.data * self.tau
        elif network == 'critic_2':
            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
                target_param._data = target_param.data * (1.0 - self.tau) + param.data * self.tau
        else:
            raise ValueError

    def hardupdate(self, network: str):
        """
        Perform a hard update of the target network parameters.

        # Arguments
            network (str): The name of the network to update ('actor' or 'critic_1' or 'critic_2').
        """
        if network == 'actor':
            for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param._data = param.data
        elif network == 'critic_1':
            for target_param, param in zip(self.critic_target_1.parameters(), self.critic_local_1.parameters()):
                target_param._data = param.data
        elif network == 'critic_2':
            for target_param, param in zip(self.critic_target_2.parameters(), self.critic_local_2.parameters()):
                target_param._data = param.data
        else:
            raise ValueError

    def predict(self, state=None):
        """
        Get the action for a given state.

        # Arguments
            state (array-like, optional): The current state.

        # Returns
            action (array-like): The predicted action for the given state.
        """
        self.actor_local.eval()
        return to_numpy(self.actor_local(to_tensor(state))).flatten()

    def train(self, batch=None, update_target=True):
        """
        Train the actor and critic networks with a given batch.

        # Arguments
            batch (tuple of array-like, optional): The batch of experiences to train on.
            update_target (bool, optional): Whether to perform a soft update of the target networks.

        # Returns
            losses (list of float): The actor and critic losses.
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
        q_next = functional.min(q_next_1, q_next_2)

        q_next = self.gamma * q_next * (1 - terminals)
        q_next.add_(rewards)
        q_next = q_next.detach()

        q_1 = self.critic_local_1(states, actions)
        q_2 = self.critic_local_2(states, actions)

        critic_1_loss = functional.mse_loss(q_1, q_next)
        critic_2_loss = functional.mse_loss(q_2, q_next)

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
        Evaluate the actor and critic networks.

        This method should be overridden in a subclass.
        """
