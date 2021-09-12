from typing import Optional

import torch
from torch import nn

from .BaseEmbeddingModel import BaseEmbeddingModel
from ..adaptors import BaseAdaptor


class LogitMatrixFactorization(BaseEmbeddingModel):
    def __init__(
        self,
        n_user: int,
        n_item: int,
        n_dim: int = 20,
        max_norm: Optional[float] = None,
        max_bias: Optional[float] = 1,
        user_embedding_init: Optional[torch.Tensor] = None,
        item_embedding_init: Optional[torch.Tensor] = None,
        user_bias_init: Optional[torch.Tensor] = None,
        item_bias_init: Optional[torch.Tensor] = None,
        user_adaptor: Optional[BaseAdaptor] = None,
        item_adaptor: Optional[BaseAdaptor] = None,
    ):
        """Set model parameters

        Args:
            n_user (int): A number of users
            n_item (int): A number of item
            n_dim (int, optional): A number of latent dimension. Defaults to 20.
            max_norm (Optional[float], optional): upper bound of norm of latent vector. Defaults to None.
            max_bias (Optional[float], optional): upper bound of bias. Defaults to 1.
            user_embedding_init (Optional[torch.Tensor], optional): initial embeddings for users. Defaults to None.
            item_embedding_init (Optional[torch.Tensor], optional): initial embeddings for item. Defaults to None.
            user_bias_init (Optional[torch.Tensor], optional): initial biases for users. Defaults to None.
            item_bias_init (Optional[torch.Tensor], optional): initial biases for item. Defaults to None.
        """

        super().__init__(
            n_user,
            n_item,
            n_dim,
            max_norm,
            user_embedding_init,
            item_embedding_init,
            user_adaptor,
            item_adaptor,
        )
        self.max_bias = max_bias
        self.weight_link = lambda x: torch.sigmoid(-x)

        if user_bias_init is None:
            self.user_bias = nn.Embedding(n_user, 1, max_norm=max_bias)

        else:
            self.user_bias = nn.Embedding.from_pretrained(
                user_bias_init.reshape(-1, 1), max_norm=max_bias
            )
            self.user_bias.weight.requires_grad = True

        if item_bias_init is None:
            self.item_bias = nn.Embedding(n_item, 1, max_norm=max_bias)

        else:
            self.item_bias = nn.Embedding.from_pretrained(
                item_bias_init.reshape(-1, 1), max_norm=max_bias
            )
            self.item_bias.weight.requires_grad = True

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> dict:
        """Method of forwarding embeddings

        Args:
            users : tensor of user indices size (n_batch).
            pos_items : tensor of item indices size (n_batch, 1)
            neg_items : tensor of item indices size (n_batch, n_neg_samples)

        Returns:
            dict: A dictionary of embeddings.
        """

        # get enmbeddigs
        embeddings_dict = {
            "user_embedding": self.user_embedding(users),
            "pos_item_embedding": self.item_embedding(pos_items),
            "neg_item_embedding": self.item_embedding(neg_items),
            "user_bias": self.user_bias(users),
            "pos_item_bias": self.item_bias(pos_items),
            "neg_item_bias": self.item_bias(neg_items)[:, :, 0],
        }

        return embeddings_dict

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """Method of predicting relevance for each pair of user and item.

        Args:
            pairs (torch.Tensor): 2d tensor which columns are [user_id, item_id].

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: inner product for each users and item pair size (n_batch).
        """

        # set users and user
        users = pairs[:, 0]
        items = pairs[:, 1]

        # get enmbeddigs
        u_emb = self.user_embedding(users)  # batch_size × dim
        i_emb = self.item_embedding(items)  # batch_size × dim

        # get bias
        u_bias = self.user_bias(users)  # batch_size × 1
        i_bias = self.item_bias(items)  # batch_size × 1

        # compute distance
        inner = torch.einsum("nd,nd->n", u_emb, i_emb)

        return inner + u_bias.reshape(-1) + i_bias.reshape(-1)

    def predict_binary(self, pairs: torch.Tensor) -> torch.Tensor:
        pred = self.predict(pairs)
        sign = torch.sign(pred)
        binary = torch.uint8((sign + 1) / 2)
        return binary

    def predict_proba(self, pairs: torch.Tensor) -> torch.Tensor:
        pred = self.predict(pairs)
        proba = torch.sigmoid(pred)
        return proba

    def get_item_score(self, users: torch.Tensor) -> torch.Tensor:
        """Method of getting scores of all items for each user.
        Args:
            users (torch.Tensor): 1d tensor of user_id size (n).

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: Tensor of item scores size (n, n_item)
        """
        u_emb = self.user_embedding(users)
        u_bias = self.user_bias(users)
        item_score = u_emb @ self.item_embedding.weight.T
        item_score += self.item_bias.weight.T + u_bias

        return item_score

    def get_item_weight(self, users: torch.Tensor) -> torch.Tensor:
        """Method of getting weight for negative sampling
        Args:
            users (torch.Tensor): 1d tensor of user_id size (n).

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: Tensor of weight size (n, n_item)
        """
        item_score = self.get_item_score(users)
        weight = self.weight_link(item_score)

        return weight
