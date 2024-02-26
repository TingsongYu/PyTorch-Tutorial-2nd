import math
import numpy as np
import warnings
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=1., last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.:
            return 0.
        return max(0., 0.5 * (1. + math.cos(math.pi * ((float(num_cycles) * progress) % 1.))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CustomDecayLR(object):
    '''
    自定义学习率变化机制
        Example:
        >>> scheduler = CustomDecayLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.epoch_step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>     validate(...)
    '''
    def __init__(self,optimizer,lr):
        self.optimizer = optimizer
        self.lr = lr

    def epoch_step(self,epoch):
        lr = self.lr
        if epoch > 12:
            lr = lr / 1000
        elif epoch > 8:
            lr = lr / 100
        elif epoch > 4:
            lr = lr / 10
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class BertLR(object):
    '''
    Bert模型内定的学习率变化机制
    Example:
        >>> scheduler = BertLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    '''
    def __init__(self,optimizer,learning_rate,t_total,warmup):
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.t_total = t_total
        self.warmup = warmup

    # 线性预热方式
    def warmup_linear(self,x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def batch_step(self,training_step):
        lr_this_step = self.learning_rate * self.warmup_linear(training_step / self.t_total,self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step

class CyclicLR(object):
    '''
    Cyclical learning rates for training neural networks
    Example:
        >>> scheduler = CyclicLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    '''
    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class ReduceLROnPlateau(object):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.


    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_acc, val_loss = validate(...)
        >>>     scheduler.epoch_step(val_loss, epoch)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=0, epsilon=1e-4, cooldown=0, min_lr=0,eps=1e-8):

        super(ReduceLROnPlateau, self).__init__()
        assert isinstance(optimizer, Optimizer)
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.monitor_op = None
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.optimizer = optimizer
        self.eps = eps
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            raise RuntimeError('Learning Rate Plateau Reducing mode %s is unknown!')
        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def reset(self):
        self._reset()

    def epoch_step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.eps:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %05d: reducing learning rate to %s.' % (epoch, new_lr))
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0

class ReduceLRWDOnPlateau(ReduceLROnPlateau):
    """Reduce learning rate and weight decay when a metric has stopped
    improving. Models often benefit from reducing the learning rate by
    a factor of 2-10 once learning stagnates. This scheduler reads a metric
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate and weight decay factor is reduced for
    optimizers that implement the the weight decay method from the paper
    `Fixing Weight Decay Regularization in Adam`_.

    .. _Fixing Weight Decay Regularization in Adam:
        https://arxiv.org/abs/1711.05101
    for AdamW or SGDW
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=1e-3)
        >>> scheduler = ReduceLRWDOnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.epoch_step(val_loss)
    """
    def epoch_step(self, metrics, epoch):
        current = metrics
        if current is None:
            warnings.warn('Learning Rate Plateau Reducing requires metrics available!', RuntimeWarning)
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    for param_group in self.optimizer.param_groups:
                        old_lr = float(param_group['lr'])
                        if old_lr > self.min_lr + self.eps:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            param_group['lr'] = new_lr
                            if self.verbose > 0:
                                print('\nEpoch %d: reducing learning rate to %s.' % (epoch, new_lr))
                        if param_group['weight_decay'] != 0:
                            old_weight_decay = float(param_group['weight_decay'])
                            new_weight_decay = max(old_weight_decay * self.factor, self.min_lr)
                            if old_weight_decay > new_weight_decay + self.eps:
                                param_group['weight_decay'] = new_weight_decay
                                if self.verbose:
                                    print('\nEpoch {epoch}: reducing weight decay factor of group {i} to {new_weight_decay:.4e}.')
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                self.wait += 1

class CosineLRWithRestarts(object):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will extend/shrink

    Example:
        >>> scheduler = CosineLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """

    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, eta_threshold=1000, verbose=False):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'],
                                 optimizer.param_groups))

        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.iteration = 0
        self.epoch_size = epoch_size
        self.eta_threshold = eta_threshold
        self.t_mult = t_mult
        self.verbose = verbose
        self.base_weight_decays = list(map(lambda group: group['weight_decay'],
                                           optimizer.param_groups))
        self.restart_period = restart_period
        self.restarts = 0
        self.t_epoch = -1
        self.batch_increments = []
        self._set_batch_increment()

    def _schedule_eta(self):
        """
        Threshold value could be adjusted to shrink eta_min and eta_max values.
        """
        eta_min = 0
        eta_max = 1
        if self.restarts <= self.eta_threshold:
            return eta_min, eta_max
        else:
            d = self.restarts - self.eta_threshold
            k = d * 0.09
            return (eta_min + k, eta_max - k)

    def get_lr(self, t_cur):
        eta_min, eta_max = self._schedule_eta()

        eta_t = (eta_min + 0.5 * (eta_max - eta_min)
                 * (1. + math.cos(math.pi *
                                  (t_cur / self.restart_period))))

        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        lrs = [base_lr * eta_t for base_lr in self.base_lrs]
        weight_decays = [base_weight_decay * eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]

        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart at epoch {}".format(self.last_epoch))
            self.restart_period *= self.t_mult
            self.restarts += 1
            self.t_epoch = 0

        return zip(lrs, weight_decays)

    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = list(np.linspace(0, 1, batches_in_epoch))

    def batch_step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self.iteration += 1
        except (IndexError):
            raise RuntimeError("Epoch size and batch size used in the "
                               "training loop and while initializing "
                               "scheduler should be the same.")

        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay


class NoamLR(object):
    '''
    主要参考论文<< Attention Is All You Need>>中的学习更新方式
    Example:
        >>> scheduler = NoamLR(d_model,factor,warm_up,optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         glopab_step += 1
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step(global_step)
        >>>     validate(...)
    '''
    def __init__(self,d_model,factor,warm_up,optimizer):
        self.optimizer = optimizer
        self.warm_up = warm_up
        self.factor = factor
        self.d_model = d_model
        self._lr = 0

    def get_lr(self,step):
        lr = self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5),step * self.warm_up ** (-1.5)))
        return lr

    def batch_step(self,step):
        '''
        update parameters and rate
        :return:
        '''
        lr = self.get_lr(step)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr
