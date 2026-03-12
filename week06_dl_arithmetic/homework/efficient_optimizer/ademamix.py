import math
 
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
 
# HINT: you may want to change these functions
def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0-a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):

    def f(beta, eps=1e-8):
        return math.log(0.5)/math.log(beta+eps)-1

    def f_inv(t):
        return math.pow(0.5, 1/(t+1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0-a) * f(beta_start) + a * f(beta_end))
    return beta_end
 
 
@torch.compile(fullgraph=True) # you can comment out this line for subtask 1
def ademamix_foreach_fn(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avg_fasts: list[Tensor],
    exp_avg_slows: list[Tensor],
    exp_avg_sqs: list[Tensor],
    bias_correction1: Tensor,
    bias_correction2: Tensor,
    alphas: Tensor,
    beta3: Tensor,
    beta1: float, beta2: float,
    lr: float, lmbda: float, eps: float,
):
    # TODO: Replace per-tensor ops with torch._foreach_* equivalents:
    # torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
    # torch._foreach_mul_(exp_avg_sqs, beta2)
    # torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)
    # etc.
    
    exp_avg_fasts = torch._foreach_lerp_(exp_avg_fasts, grads, 1 - beta1)

    exp_avg_slows = torch._foreach_lerp_(exp_avg_slows, grads, 1 - beta3)
    exp_avg_sqs = torch._foreach_mul_(exp_avg_sqs, beta2)
    exp_avg_sqs = torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

    denoms = torch._foreach_add_(torch._foreach_div(torch._foreach_sqrt(exp_avg_sqs), torch.sqrt(bias_correction2)), eps)
    updates = torch._foreach_div(torch._foreach_add_(torch._foreach_div(exp_avg_fasts, bias_correction1), torch._foreach_mul(exp_avg_slows, alphas)), denoms)

    updates = torch._foreach_add_(updates, torch._foreach_mul(params, lmbda))
    params = torch._foreach_add_(params, torch._foreach_mul(updates, -lr))

 
class AdEMAMix(Optimizer):
    r"""Implements the AdEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999, 0.9999)) 
            corresponding to beta_1, beta_2, beta_3 in AdEMAMix
        alpha (float): AdEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta3_warmup (int, optional): number of warmup steps used to increase beta3 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=2.0, 
                 beta3_warmup=None, alpha_warmup=None,  eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(lr=lr, betas=betas, eps=eps, alpha=alpha, beta3_warmup=beta3_warmup,
                        alpha_warmup=alpha_warmup, weight_decay=weight_decay)
        super(AdEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            
            lr = group["lr"]
            lmbda = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]
 
            params: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avg_fasts: list[Tensor] = []
            exp_avg_slows: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            bias_correction1: Tensor = 0.0
            bias_correction2: Tensor = 0.0
            alphas: Tensor = 0.0
            beta3: Tensor = 0.0

            # TODO: prepare lists of tensors for 'foreach' step
            # as well as all other necessary inputs
            # HINT: (think about input types)
            
            for p in group['params']:
                params.append(p)
                grads.append(p.grad)
                
                if self.state[p]:
                    exp_avg_fasts.append(self.state[p]['exp_avg_fast'])
                    exp_avg_slows.append(self.state[p]['exp_avg_slow'])
                    exp_avg_sqs.append(self.state[p]['exp_avg_sq'])
                else:
                    exp_avg_fasts.append(torch.zeros_like(p, memory_format=torch.preserve_format))
                    exp_avg_slows.append(torch.zeros_like(p, memory_format=torch.preserve_format))
                    exp_avg_sqs.append(torch.zeros_like(p, memory_format=torch.preserve_format))
                    self.state[p]['step'] = 0
                self.state[p]['step'] += 1
                bias_correction1 = 1 - beta1 ** self.state[p]['step']
                bias_correction2 = 1 - beta2 ** self.state[p]['step']
                
                if alpha_warmup is not None:
                    alphas = linear_warmup_scheduler(self.state[p]['step'], alpha_end=alpha_final, alpha_start=0, warmup=alpha_warmup)
                else:
                    alphas = alpha_final

                if beta3_warmup is not None:
                    beta3 = linear_hl_warmup_scheduler(self.state[p]['step'], beta_end=beta3_final, beta_start=beta1, warmup=beta3_warmup)
                else:
                    beta3 = beta3_final
            
            if params[0].grad is None:
                continue

            ademamix_foreach_fn(
                params=params,
                grads=grads,
                exp_avg_fasts=exp_avg_fasts,
                exp_avg_slows=exp_avg_slows,
                exp_avg_sqs=exp_avg_sqs,
                bias_correction1=torch.tensor(bias_correction1),
                bias_correction2=torch.tensor(bias_correction2),
                alphas=torch.tensor(alphas),
                beta3=torch.tensor(beta3),
                beta1=beta1,
                beta2=beta2,
                lr=lr, lmbda=lmbda, eps=eps,
            )

            for i, p in enumerate(group['params']):
                self.state[p]['exp_avg_fast'] = exp_avg_fasts[i]
                self.state[p]['exp_avg_slow'] = exp_avg_slows[i]
                self.state[p]['exp_avg_sq'] = exp_avg_sqs[i]

        return loss