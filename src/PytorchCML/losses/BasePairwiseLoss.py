from typing import Optional

import torch
from torch import nn


class BasePairwiseLoss(nn.Module):
    """ Class of abstract loss module for pairwise loss like matrix factorization. 
    """

    def forward(self, user_emb: torch.Tensor,
                pos_item_emb: torch.Tensor,
                neg_item_emb: torch.Tensor,
                user_bias: torch.Tensor,
                pos_item_bias: torch.Tensor,
                neg_item_bias: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            user_emb : embeddings of user size (n_batch, d)
            pos_item_emb : embeddings of positive item size (n_batch, d)
            neg_item_emb : embeddings of negative item size (n_batch, n_neg_samples, d)
            user_bias : bias of user size (n_batch, 1)
            pos_item_bias : bias of positive item size (n_batch, 1)
            neg_item_bias : bias of negative item size (n_batch, n_neg_samples, 1)

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [description]
        """

        raise NotImplementedError
