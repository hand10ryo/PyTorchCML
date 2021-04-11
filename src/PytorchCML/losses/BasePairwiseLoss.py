from typing import Optional

import torch
from torch import nn


class BasePairwiseLoss(nn.Module):

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

        raise NotImplementedError
