from typing import Optional

import torch
from torch import nn


class BaseEmbeddingModel(nn.Module):
    """ Class of abstract embeddings model getting embedding or predict relevance from indices.
    """

    def __init__(self, n_user: int, n_item: int, n_dim: int = 20, max_norm: Optional[float] = 1,
                 user_embedding_init: Optional[torch.Tensor] = None,
                 item_embedding_init: Optional[torch.Tensor] = None):
        """ Set embeddings.

        Args:
            n_user (int): A number of users.
            n_item (int): A number of items.
            n_dim (int, optional): A number of dimention of embeddings. Defaults to 20.
            max_norm (Optional[float], optional): Allowed maximum norm. Defaults to 1.
            user_embedding_init (Optional[torch.Tensor], optional): Initial user embeddings. Defaults to None.
            item_embedding_init (Optional[torch.Tensor], optional): Initial item embeddings. Defaults to None.
        """
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

    def forward(self, users: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
