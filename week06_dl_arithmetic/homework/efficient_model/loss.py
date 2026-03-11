"""
Cross Entropy Loss for Causal LM
"""

import torch
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


class CrossEntropyLoss(nn.Module):
    """Fused Linear Cross Entropy for causal LM."""
    # TODO: Replace with fused linear cross entropy (LigerFusedLinearCrossEntropyLoss)
    # The fused version takes hidden_states + lm_head.weight instead of logits

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=ignore_index)


    def forward(self, hidden_states, lm_head_weight, labels) -> torch.Tensor:
        # TODO: Implement forward pass
        bs, seq_len, hidden_dim = hidden_states.shape
        return self.loss_fn(lin_weight=lm_head_weight, _input=hidden_states.reshape(bs * seq_len, hidden_dim), target=labels.reshape(-1))
