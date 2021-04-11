from typing import Optional

import torch
from torch import nn

from .BaseEmbeddingModel import BaseEmbeddingModel


class LogitMatrixFactorization(BaseEmbeddingModel):

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Args:
            users : tensor of user indices size (n_batch). 
            items : tensor of item indices 
                   pos_pairs -> size (n_batch, 1), 
                   neg_pairs -> size (n_batch, n_neg_samples)


        Returns:
            inner : inner product for each users and item pair
                   pos_pairs -> size (n_batch, 1), 
                   neg_pairs -> size (n_batch, n_neg_samples)

        """

        # get enmbeddigs
        u_emb = self.user_embedding(users)  # batch_size × dim
        i_emb = self.item_embedding(items)  # batch_size × n_samples × dim

        # compute inner product
        inner = torch.einsum('nd,njd->nj', u_emb, i_emb)

        return inner

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairs : tensor of indices for user and item pairs size (n_pairs, 2).
        Returns:
            inner : inner product for each users and item pair size (n_batch)
        """
        # set users and user
        users = pairs[:, 0]
        items = pairs[:, 1]

        # get enmbeddigs
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)

        # compute distance
        inner = torch.einsum('nd,nd->n', u_emb, i_emb)

        return inner

    def predict_binary(self, pairs: torch.Tensor) -> torch.Tensor:
        inner = self.predict(pairs)
        sign = torch.sign(inner)
        binary = torch.uint8((sign + 1) / 2)
        return binary

    def predict_proba(self, pairs: torch.Tensor) -> torch.Tensor:
        inner = self.predict(pairs)
        proba = torch.sigmoid(inner)
        return proba
