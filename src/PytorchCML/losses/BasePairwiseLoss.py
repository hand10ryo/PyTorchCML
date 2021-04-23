from typing import Optional

import torch
from torch import nn


class BasePairwiseLoss(nn.Module):
    """ Class of abstract loss module for pairwise loss like matrix factorization. 
    """

    def forward(self,
                pos_inner: torch.Tensor,
                neg_inner: torch.Tensor,
                sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ 

        Args:
            Args:
            pos_inner (torch.Tensor): inner product of pos pairs of size (n_batch, 1)
            neg_inner (torch.Tensor): inner product of neg pairs of size (n_batch, n_neg_samples)
            sample_weight (Optional[torch.Tensor], optional): sample weight size (n_batch). Defaults to None.

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [description]
        """

        raise NotImplementedError
