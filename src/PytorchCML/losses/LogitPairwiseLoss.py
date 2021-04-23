from typing import Optional

import torch
from torch import nn

from .BasePairwiseLoss import BasePairwiseLoss


class LogitPairwiseLoss(BasePairwiseLoss):
    """ Class of pairwise logit loss for Logistic Matrix Factorization 
    """

    def __init__(self):
        super().__init__()
        self.LogSigmoid = nn.LogSigmoid()

    def forward(self,
                pos_inner: torch.Tensor,
                neg_inner: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pos_inner : inner product of pos pairs of size (n_batch, 1)
            neg_inner : inner product of neg pairs of size (n_batch, n_neg_samples)
            sample_weight : sample weight size (n_batch)
        """
        n_batch = pos_inner.shape[0]
        n_pos = 1
        n_neg = neg_inner.shape[1]
        pos_loss = - nn.LogSigmoid()(pos_inner).sum()
        neg_loss = - nn.LogSigmoid()(-neg_inner).sum()

        loss = (pos_loss + neg_loss) / (n_batch * (n_pos + n_neg))
        return loss
