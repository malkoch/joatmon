from joatmon.nn import functional as f
from joatmon.nn.core import Optimizer

__all__ = ['Adam']


class Adam(Optimizer):
    """
    Implements the Adam optimization algorithm.

    Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure
    to update network weights iterative based on training data.

    # Attributes
        params (iterable): An iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): The learning rate. Default is 1e-3.
        betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
        eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.
        amsgrad (bool, optional): Whether to use the AMSGrad variant of this algorithm. Default is False.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        """
        Initializes the Adam class.

        # Arguments
            params (iterable): An iterable of parameters to optimize or dicts defining parameter groups.
            lr (float, optional): The learning rate. Default is 1e-3.
            betas (Tuple[float, float], optional): Coefficients used for computing running averages of gradient and its square. Default is (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Default is 1e-8.
            weight_decay (float, optional): Weight decay (L2 penalty). Default is 0.
            amsgrad (bool, optional): Whether to use the AMSGrad variant of this algorithm. Default is False.
        """
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= eps:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        """
        Sets the state of the optimizer.

        # Arguments
            state (dict): The state of the optimizer.
        """
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self):
        """
        Performs a single optimization step.

        This function is called once per optimization step to update the parameters.
        """
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                # print(p, p.grad, p.requires_grad)
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = f.zeros_like(p)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = f.zeros_like(p)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = f.zeros_like(p)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            f.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group['amsgrad'],
                beta1,
                beta2,
                group['lr'],
                group['weight_decay'],
                group['eps'],
            )
