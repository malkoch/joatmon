import warnings

from joatmon.nn.core import LRScheduler

__all__ = ['ExponentialLR']


class ExponentialLR(LRScheduler):
    """
    Implements the Exponential Learning Rate Scheduler.

    This scheduler decays the learning rate of each parameter group by a specified gamma every epoch.

    # Attributes
        optimizer (Optimizer): The optimizer for which the learning rate will be scheduled.
        gamma (float): The multiplicative factor of learning rate decay.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
        verbose (bool, optional): If True, prints a message to stdout for each update. Default is False.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        """
        Initializes the ExponentialLR class.

        # Arguments
            optimizer (Optimizer): The optimizer for which the learning rate will be scheduled.
            gamma (float): The multiplicative factor of learning rate decay.
            last_epoch (int, optional): The index of the last epoch. Default is -1.
            verbose (bool, optional): If True, prints a message to stdout for each update. Default is False.
        """
        self.gamma = gamma
        super(ExponentialLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Computes the learning rate for the current epoch.

        # Returns
            list: The learning rates for each parameter group.
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                'To get the last learning rate computed by the scheduler, please use `get_last_lr()`.', UserWarning
            )

        if self.last_epoch == 0:
            return self.base_lrs
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        """
        Computes the learning rate for the current epoch in closed form.

        # Returns
            list: The learning rates for each parameter group.
        """
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]
