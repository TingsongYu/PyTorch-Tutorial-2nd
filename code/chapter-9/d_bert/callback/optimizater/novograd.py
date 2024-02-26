import torch
import math
from torch.optim.optimizer import Optimizer


class NovoGrad(Optimizer):
    """Implements NovoGrad algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    Example:
        >>> model = ResNet()
        >>> optimizer = NovoGrad(model.parameters(), lr=1e-2, weight_decay=1e-5)
    """

    def __init__(self, params, lr=0.01, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0, grad_averaging=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NovoGrad does not support sparse gradients')
                state = self.state[p]
                g_2 = torch.sum(grad ** 2)
                if len(state) == 0:
                    state['step'] = 0
                    state['moments'] = grad.div(g_2.sqrt() + group['eps']) + \
                                       group['weight_decay'] * p.data
                    state['grads_ema'] = g_2
                moments = state['moments']
                grads_ema = state['grads_ema']
                beta1, beta2 = group['betas']
                state['step'] += 1
                grads_ema.mul_(beta2).add_(1 - beta2, g_2)

                denom = grads_ema.sqrt().add_(group['eps'])
                grad.div_(denom)
                # weight decay
                if group['weight_decay'] != 0:
                    decayed_weights = torch.mul(p.data, group['weight_decay'])
                    grad.add_(decayed_weights)

                # Momentum --> SAG
                if group['grad_averaging']:
                    grad.mul_(1.0 - beta1)

                moments.mul_(beta1).add_(grad)  # velocity

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.add_(-step_size, moments)

        return loss
