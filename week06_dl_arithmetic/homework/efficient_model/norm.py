"""
Zero-Centered RMSNorm
"""

import torch
import torch.nn as nn


@torch.compile
def rmsnorm_forward(x, weight, eps):
    """Zero-Centered RMSNorm forward."""
    # TODO: Replace with fused implementation
    input_dtype = x.dtype
    x = x.float()
    x_squared = x * x
    mean_squared = x_squared.mean(dim=-1, keepdim=True)
    mean_squared_eps = mean_squared + eps
    rsqrt = torch.rsqrt(mean_squared_eps)
    normalized = x * rsqrt
    scale = 1.0 + weight.float()
    output = normalized * scale
    # TODO: Think about additional return parameters
    return output.to(input_dtype), rsqrt.to(input_dtype)


@torch.compile
def rmsnorm_backward(grad_output, x, weight, eps, rsqrt):
    """Zero-Centered RMSNorm backward."""
    input_dtype = x.dtype
    x = x.float()
    weight = weight.float()
    
    w1 = 1.0 + weight.float()
    result = grad_output * w1 * rsqrt - (grad_output * x * w1).mean(dim=-1, keepdim=True) * x * rsqrt ** 3

    normalized = x * rsqrt
    dl_dw = (grad_output * normalized).sum(dim=0)
    return result.to(input_dtype), dl_dw.to(input_dtype)


class RMSNormFunction(torch.autograd.Function):
    """
    Template for memory-efficient and fused Zero-Centered RMSNorm autograd function.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        # TODO: Replace with fused implementation
        output, rsqrt = rmsnorm_forward(x, weight, eps)

        # TODO: Save tensors for backward (make it memory-efficient)
        ctx.save_for_backward(x, weight, rsqrt)  # TODO: Fill this
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement fused backward pass
        # TODO: Make it work with memory-efficient forward
        x, weight, rsqrt = ctx.saved_tensors
        eps = ctx.eps
        grad_output, dl_dw = rmsnorm_backward(grad_output, x, weight, eps, rsqrt)
        return grad_output, dl_dw, None


class RMSNorm(nn.Module):
    """
    Zero-Centered RMSNorm: y = x/rms(x) * (1 + weight), weight init to zeros.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)
