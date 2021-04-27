from typing import Optional

import torch
from torch import nn

from .BasePairwiseLoss import BasePairwiseLoss


class LogitPairwiseLoss(BasePairwiseLoss):
    """ Class of pairwise logit loss for Logistic Matrix Factorization 
    """

    def __init__(self, regularizers: list = []):
        super().__init__(regularizers)
        self.LogSigmoid = nn.LogSigmoid()

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
        """
        embeddings_dict = {
            "user_emb": user_emb,  # (B, d)
            "pos_item_emb": pos_item_emb,  # (B, d)
            "neg_item_emb": neg_item_emb,  # (B, N, d)
            "user_bias": user_bias,  # (B, 1)
            "pos_item_bias": pos_item_bias,  # (B, 1)
            "neg_item_bias": neg_item_bias  # (B, N, 1)
        }

        n_batch = user_emb.shape[0]
        n_pos = 1
        n_neg = neg_item_emb.shape[1]
        neg_item_bias = neg_item_bias.reshape(n_batch, n_neg)

        pos_inner = torch.einsum('nd,nd->n', user_emb, pos_item_emb)
        neg_inner = torch.einsum('nd,njd->nj', user_emb, neg_item_emb)

        pos_y_hat = pos_inner + (user_bias + pos_item_bias).reshape(-1)
        neg_y_hat = neg_inner + user_bias + neg_item_bias

        pos_loss = - nn.LogSigmoid()(pos_y_hat).sum()
        neg_loss = - nn.LogSigmoid()(-neg_y_hat).sum()

        loss = (pos_loss + neg_loss) / (n_batch * (n_pos + n_neg))
        reg = self.regularize(embeddings_dict)

        return loss + reg
