import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_var, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        
        input_sum = input.sum(dim=0)[None, ...]
        input_sum_sq = (input ** 2).sum(dim=0)[None, ...]
        
        total = torch.concat([input_sum, input_sum_sq], dim=0)
        
        result_shape = tuple([dist.get_world_size()] + list(total.shape))
        total_list = torch.zeros(result_shape, dtype=total.dtype, device=input.device)
        dist.all_gather_into_tensor(total_list, total)
        total_sum = total_list[:, 0].sum(dim=0)
        total_sum_sq = total_list[:, 1].sum(dim=0)
        
        N = input.shape[0] * dist.get_world_size()

        cur_mean = total_sum / N
        cur_var = total_sum_sq / N - cur_mean ** 2

        running_mean.mul_(momentum).add_(cur_mean, alpha=1 - momentum)
        running_var.mul_(momentum).add_(cur_var, alpha=1 - momentum)

        output = (input - cur_mean) / (cur_var + eps) ** 0.5
        ctx.save_for_backward(input, cur_mean, cur_var, output)
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        
        input, cur_mean, cur_var, output = ctx.saved_tensors
        eps = ctx.eps

        grad_sum = grad_output.sum(dim=0)[None, ...]
        grad_x_output_sum = (grad_output * output).sum(dim=0)[None, ...]
        
        total = torch.concat([grad_sum, grad_x_output_sum], dim=0)
        result_shape = tuple([dist.get_world_size()] + list(total.shape))
        total_list = torch.zeros(result_shape, dtype=total.dtype, device=input.device)
        dist.all_gather_into_tensor(total_list, total)

        total_sum_grad = total_list[:, 0].sum(dim=0)
        total_sum_grad_x_output = total_list[:, 1].sum(dim=0)

        N = input.shape[0] * dist.get_world_size()
        dx = 1/N/(cur_var + eps) ** 0.5 * (N * grad_output - total_sum_grad - output * total_sum_grad_x_output)

        return dx, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        # your code here
        # self.running_mean = torch.zeros((num_features,))
        # self.running_var = torch.ones((num_features,))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        
        if not self.training:
            return (input - self.running_mean) / (self.running_var + self.eps) ** 0.5
        
        return sync_batch_norm.apply(input, self.running_mean, self.running_var, self.eps, self.momentum)
