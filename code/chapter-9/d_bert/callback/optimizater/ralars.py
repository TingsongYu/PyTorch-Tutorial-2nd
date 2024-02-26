import math
import torch
from torch.optim.optimizer import Optimizer


class RaLars(Optimizer):
    """Implements the RAdam optimizer from https://arxiv.org/pdf/1908.03265.pdf
    with optional Layer-wise adaptive Scaling from https://arxiv.org/pdf/1708.03888.pdf

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate
        betas (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        scale_clip (float, optional): the maximal upper bound for the scale factor of LARS
    Example:
        >>> model = ResNet()
        >>> optimizer = RaLars(model.parameters(), lr=0.001)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 scale_clip=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RaLars, self).__init__(params, defaults)
        # LARS arguments
        self.scale_clip = scale_clip
        if self.scale_clip is None:
            self.scale_clip = (0, 10)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # Get group-shared variables
            beta1, beta2 = group['betas']
            sma_inf = group.get('sma_inf')
            # Compute max length of SMA on first step
            if not isinstance(sma_inf, float):
                group['sma_inf'] = 2 / (1 - beta2) - 1
                sma_inf = group.get('sma_inf')

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute length of SMA
                sma_t = sma_inf - 2 * state['step'] * (1 - bias_correction2) / bias_correction2

                update = torch.zeros_like(p.data)
                if sma_t > 4:
                    #  Variance rectification term
                    r_t = math.sqrt((sma_t - 4) * (sma_t - 2) * sma_inf / ((sma_inf - 4) * (sma_inf - 2) * sma_t))
                    #  Adaptive momentum
                    update.addcdiv_(r_t, exp_avg / bias_correction1,
                                    (exp_avg_sq / bias_correction2).sqrt().add_(group['eps']))
                else:
                    # Unadapted momentum
                    update.add_(exp_avg / bias_correction1)

                # Weight decay
                if group['weight_decay'] != 0:
                    update.add_(group['weight_decay'], p.data)

                # LARS
                p_norm = p.data.pow(2).sum().sqrt()
                update_norm = update.pow(2).sum().sqrt()
                phi_p = p_norm.clamp(*self.scale_clip)
                # Compute the local LR
                if phi_p == 0 or update_norm == 0:
                    local_lr = 1
                else:
                    local_lr = phi_p / update_norm

                state['local_lr'] = local_lr

                p.data.add_(-group['lr'] * local_lr, update)

        return loss
