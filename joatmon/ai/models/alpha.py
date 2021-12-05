import os

import numpy as np
import torch
import torch.optim as optim

from joatmon.ai.core import CoreModel
from joatmon.ai.utility import (
    load,
    save
)

__all__ = ['AlphaModel']


class AlphaModel(CoreModel):
    """
    Alpha Zero

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

    def __init__(self, lr=1e-3, tau=1e-4, gamma=0.99, net=None):
        super(AlphaModel, self).__init__()

        self.net = net

        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.cuda = torch.cuda.is_available()

        if self.cuda:
            self.net = self.net.cuda()

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.initialize()

    def load(self, path=''):
        load(self.net, os.path.join(path, 'net'))

    def save(self, path=''):
        save(self.net, os.path.join(path, 'net'))

    def initialize(self, w_init=None, b_init=None):
        for module in self.net.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear)):  # batch norm will be different
                if w_init is not None:
                    torch.nn.init.kaiming_normal(module.weight)
                if b_init is not None:  # bias init will be different
                    torch.nn.init.kaiming_normal(module.bias)

    def predict(self, state=None):
        state = torch.FloatTensor(state.astype(np.float32))
        if self.cuda:
            state = state.contiguous().cuda()
        self.net.eval()
        with torch.no_grad():
            pi, v = self.net(state)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def train(self, batch=None, update_target=True):
        self.net.train()

        boards, pis, vs = batch
        boards = torch.FloatTensor(np.array(boards).astype(np.float32))
        target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
        target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

        # predict
        if self.cuda:
            boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

        self.optimizer.zero_grad()
        # print('boards:', boards.size(), 'target_pis:', target_pis.size(), 'target_vs:', target_vs.size())

        # compute output
        out_pi, out_v = self.net(boards)
        # l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
        # l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
        # total_loss = l_pi + l_v
        # print('out_pi:', out_pi, 'out_v:', out_v[:, 0])

        l_v = (target_vs - out_v[:, 0]) ** 2
        # print('l_v:', l_v)
        l_pi = torch.sum((-target_pis * (1e-6 + out_pi).log()), 1)
        # print('l_pi:', l_pi)
        total_loss = (l_v.view(-1).float() + l_pi).mean()
        # print('total_loss:', total_loss)

        # compute gradient and do SGD step
        total_loss.backward()
        self.optimizer.step()

        return [l_pi.mean().item(), l_v.mean().item(), total_loss.item()]

    def evaluate(self):
        pass
