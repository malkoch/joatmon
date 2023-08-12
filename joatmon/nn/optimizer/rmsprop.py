from joatmon.nn import functional as f
from joatmon.nn.core import Optimizer

__all__ = ['RMSprop']


class RMSprop(Optimizer):
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

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= momentum:
            raise ValueError('Invalid momentum value: {}'.format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError('Invalid alpha value: {}'.format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            alphas = []
            momentum_buffers = []
            grad_avgs = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['square_avg'] = f.zeros_like(p)
                        if group['momentum'] > 0:
                            state['momentum_buffer'] = f.zeros_like(p)
                        if group['centered']:
                            state['grad_avg'] = f.zeros_like(p)

                    square_avgs.append(state['square_avg'])
                    alphas.append(group['alpha'])

                    if group['momentum'] > 0:
                        momentum_buffers.append(state['momentum_buffer'])
                    if group['centered'] > 0:
                        grad_avgs.append(state['grad_avg'])

                    state['step'] += 1
            f.rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                alphas,
                momentum_buffers,
                grad_avgs,
                group['momentum'],
                group['centered'],
                group['lr'],
                group['weight_decay'],
                group['eps'],
            )
