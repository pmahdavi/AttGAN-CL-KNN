#### The code is adopted from https://github.com/zeke-xie/Positive-Negative-Momentum/blob/main/pnm_optim/adapnm.py

import math
import torch
from torch.optim.optimizer import Optimizer, required

class my_adam(Optimizer): 
    
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999 ), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0. <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(my_adam, self).__init__(params, defaults)
        

    @torch.no_grad()
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('This optimizer does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)



                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq = state['exp_avg_sq']
                exp_avg = state['exp_avg']
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss

class my_adam_nm(Optimizer): 
    
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999 , 0.1 , 0.05), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0. <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(my_adam_nm, self).__init__(params, defaults)
        

    @torch.no_grad()
    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('This optimizer does not support sparse gradients.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['last'] = torch.zeros_like(p)



                beta1, beta2, beta3, beta4 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg_sq = state['exp_avg_sq']
                exp_avg = state['exp_avg']
                
                exp_avg.mul_(beta1).add_(grad, alpha = beta3).add_(state['last'], alpha = - beta4 )
                
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                state['last'] = grad
                
        return loss


class SGD_nm(Optimizer):
    r"""Implements Positive-Negative Momentum (PNM).
    It has be proposed in 
    `Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve 
    Generalization`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): inertia coefficients used for computing
            pn momentum(default: (0.9, 1.))
        weight_decay (float, optional): weight decay (default: 0)
        decoupled (bool, optional): decoupled weight decay or L2 regularization (default: True)
    """

    def __init__(self, params, lr=required, betas=(0.9, 1., 0.1)):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not  0. <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        defaults = dict(lr=lr, betas=betas)
        super(SGD_nm, self).__init__(params, defaults)



    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad

                # Perform decoupled weight decay or L2 Regularization
                #if group['decoupled']:
                #    p.mul_(1 - group['lr'] * group['weight_decay'])
                #else:
                #    d_p.add_(p.data, alpha=group['weight_decay'])

                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['last'] = torch.zeros_like(p)
                    state['og_momentum'] = torch.zeros_like(p)
                    
                state['step'] += 1
                og_momentum = state['og_momentum']

                    
                og_momentum.mul_(beta1).add_(d_p, alpha= beta2).add_(state['last'], alpha = - beta3)

            
                p.add_(og_momentum, alpha=-group['lr'])
                state['last'] = d_p
        return loss