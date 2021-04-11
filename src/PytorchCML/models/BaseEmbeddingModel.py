from typing import Optional

import torch
from torch import nn


class BaseEmbeddingModel(nn.Module):
    def __init__(self, n_user: int, n_item: int, n_dim: int = 20, max_norm: Optional[float] = 1,
                 user_embedding_init: Optional[torch.Tensor] = None,
                 item_embedding_init: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_dim = n_dim
        self.max_norm = max_norm

        if user_embedding_init is None:
            self.user_embedding = nn.Embedding(
                n_user, n_dim, sparse=False, max_norm=max_norm)

        else:
            self.user_embedding = nn.Embedding.from_pretrained(
                user_embedding_init)
            self.user_embedding.weight.requires_grad = True

        if item_embedding_init is None:
            self.item_embedding = nn.Embedding(
                n_item, n_dim, sparse=False, max_norm=max_norm)
        else:
            self.item_embedding = nn.Embedding.from_pretrained(
                item_embedding_init)
            self.item_embedding.weight.requires_grad = True

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CollaborativeMetricLearning(BaseEmbeddingModel):

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Args:
            users : tensor of user indices size (n_batch, 1). 
            items : tensor of item indices 
                   pos_pairs -> size (n_batch, 1), 
                   neg_pairs -> size (n_batch, n_neg_samples)


        Returns:
            dist : distance for each users and item pair
                   pos_pairs -> size (n_batch, 1, 1), 
                   neg_pairs -> size (n_batch, 1, n_neg_samples)

        """

        # get enmbeddigs
        u_emb = self.user_embedding(users)  # batch_size × 1 × dim
        i_emb = self.item_embedding(items)  # batch_size × n_samples × dim

        # compute distance
        dist = torch.cdist(u_emb, i_emb)  # batch_size × n_samples × 1

        return dist

    def spreadout_distance(self, pos_items: torch.Tensor, neg_itmes: torch.Tensor):
        """
         Args:
            pos_items : tensor of user indices size (n_batch, 1). 
            neg_itmes : tensor of item indices size (n_neg_candidates, 1)
        """

        # get enmbeddigs
        pos_i_emb = self.item_embedding(pos_items)  # n_batch × 1 × dim
        neg_i_emb = self.item_embedding(neg_itmes)  # n_neg_candidates ×　dim

        # coumute dot product
        prod = torch.einsum("nid,md->nm", pos_i_emb, neg_i_emb)

        return prod

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pairs : tensor of indices for user and item pairs size (n_pairs, 2).
        Returns:
            dist : distance for each users and item pair size (n_batch, 1, 1)
        """
        # set users and user
        users = pairs[:, :1]
        items = pairs[:, 1:2]

        # get enmbeddigs
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)

        # compute distance
        dist = torch.cdist(u_emb, i_emb).reshape(-1)

        return self.max_norm * 2 - dist
