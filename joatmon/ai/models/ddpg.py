import copy
import os

from joatmon.ai.models.core import CoreModel
from joatmon.ai.network.ddpg import (
    DDPGActor,
    DDPGCritic
)
from joatmon.ai.utility import (
    load,
    save,
    to_numpy,
    to_tensor
)
from joatmon.nn import init
from joatmon.nn.layer.batchnorm import BatchNorm
from joatmon.nn.layer.conv import Conv
from joatmon.nn.layer.linear import Linear
from joatmon.nn.loss.huber import HuberLoss
from joatmon.nn.optimizer.adam import Adam

__all__ = ['DDPGModel']


class DDPGModel(CoreModel):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`nn.Model` instance): See [Model](#) for details.
        critic_model (`nn.Model` instance): See [Model](#) for details.
        optimizer (`optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`layers.Input` / `layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, lr=1e-3, tau=1e-4, gamma=0.99, in_features=1, out_features=1):
        super(DDPGModel, self).__init__()

        self.actor_local = DDPGActor(in_features, out_features)
        self.actor_target = copy.deepcopy(self.actor_local)
        self.hardupdate('actor')

        self.critic_local = DDPGCritic(in_features, out_features)
        self.critic_target = copy.deepcopy(self.critic_local)
        self.hardupdate('critic')

        self.lr = lr
        self.tau = tau
        self.gamma = gamma

        self.actor_optimizer = Adam(list(self.actor_local.parameters()), lr=self.lr)

        self.critic_optimizer = Adam(list(self.critic_local.parameters()), lr=self.lr)
        self.critic_loss = HuberLoss()

    def load(self, path=''):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        load(self.actor_local, os.path.join(path, 'actor_local'))
        load(self.actor_target, os.path.join(path, 'actor_target'))
        load(self.critic_local, os.path.join(path, 'critic_local'))
        load(self.critic_target, os.path.join(path, 'critic_target'))

    def save(self, path=''):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        save(self.actor_local, os.path.join(path, 'actor_local'))
        save(self.actor_target, os.path.join(path, 'actor_target'))
        save(self.critic_local, os.path.join(path, 'critic_local'))
        save(self.critic_target, os.path.join(path, 'critic_target'))

    def initialize(self, w_init=None, b_init=None):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
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
        for module in self.critic_local.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)
        for module in self.critic_target.modules():
            if isinstance(
                    module, (Conv, BatchNorm, Linear)
            ):  # batch norm will be different
                if w_init is not None:
                    init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    init.kaiming_normal(module.bias)

    def softupdate(self, network: str):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        if network == 'actor':
            for target_param, param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
                target_param._data = target_param.data * (1.0 - self.tau) + param.data * self.tau
        elif network == 'critic':
            for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                target_param._data = target_param.data * (1.0 - self.tau) + param.data * self.tau
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
                target_param._data = param.data
        elif network == 'critic':
            for target_param, param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
                target_param._data = param.data
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
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
