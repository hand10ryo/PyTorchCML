from typing import Optional

import torch
from torch import nn

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from ..adaptors import BaseAdaptor


class BaseEmbeddingModel(nn.Module):
    """Class of abstract embeddings model getting embedding or predict relevance from indices."""

    def __init__(
        self,
        n_user: int,
        n_item: int,
        n_dim: int = 20,
        max_norm: Optional[float] = 1,
        user_embedding_init: Optional[torch.Tensor] = None,
        item_embedding_init: Optional[torch.Tensor] = None,
        user_adaptor: Optional[BaseAdaptor] = None,
        item_adaptor: Optional[BaseAdaptor] = None,
    ):
        """Set embeddings.

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
        self.user_adaptor = user_adaptor
        self.item_adaptor = item_adaptor

        if user_embedding_init is None:
            self.user_embedding = nn.Embedding(
                n_user, n_dim, sparse=False, max_norm=max_norm
            )

        else:
            self.user_embedding = nn.Embedding.from_pretrained(
                user_embedding_init)
            self.user_embedding.weight.requires_grad = True

        if item_embedding_init is None:
            self.item_embedding = nn.Embedding(
                n_item, n_dim, sparse=False, max_norm=max_norm
            )
        else:
            self.item_embedding = nn.Embedding.from_pretrained(
                item_embedding_init)
            self.item_embedding.weight.requires_grad = True

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> dict:
        """Method of forwarding which returns embeddings

        Args:
            users (torch.Tensor): Tensor of indices of user.
            pos_items (torch.Tensor): Tensor of indices of positive items.
            neg_items (torch.Tensor): Tensor of indices of negative items.

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        raise NotImplementedError

    def predict(self, pairs: torch.Tensor) -> torch.Tensor:
        """Method of predicting relevance for each pair of user and item.

        Args:
            pairs (torch.Tensor): Tensor whose columns are [user_id, item_id]

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: Tensor of relevance size (pairs.shape[0])
        """
        raise NotImplementedError

    def get_item_score(self, users: torch.Tensor) -> torch.Tensor:
        """Method of getting scores of all items for each user.
        Args:
            users (torch.Tensor): 1d tensor of user_id size (n).

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: Tensor of item scores size (n, n_item)
        """
        raise NotImplementedError

    def get_item_weight(self, users: torch.Tensor) -> torch.Tensor:
        """Method of getting weight for negative sampling
        Args:
            users (torch.Tensor): 1d tensor of user_id size (n).

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: Tensor of weight size (n, n_item)
        """
        raise NotImplementedError

    def get_topk_items(self, users: torch.Tensor, k: int, num_batch: int = 100, n_jobs: int = -1):
        """Method of getting top k items for for each user.
        Args:
            users (torch.Tensor): 1d tensor of user_id size (n).
            k : number of top items.
            num_batch : number of users for a batch.
            n_job : number of using process.

        Returns:
            torch.Tensor: Tensor of item scores size (n, n_item)
        """

        batches = torch.split(users, num_batch)
        inputs = tqdm(batches)
        items = torch.LongTensor(torch.arange(self.n_item))

        def predict_user(i, batch_users, k):
            users_expand = batch_users.expand(self.n_item, -1).T.reshape(-1, 1)
            items_expand = items.expand(len(batch_users), -1).reshape(-1, 1)
            pairs_tensor = torch.cat([users_expand, items_expand], axis=1)
            pairs_array = pairs_tensor.cpu().detach().numpy()
            pairs_df = pd.DataFrame(pairs_array, columns=['user', 'item'])
            score_tensor = self.predict(pairs_tensor)
            pairs_df['score'] = score_tensor.cpu().detach().numpy()
            pairs_df = pairs_df.sort_values(
                by=["user", "score"], ascending=[True, False])
            topk_pairs = pairs_df.groupby("user").head(k)
            return i, topk_pairs

        scored = Parallel(n_jobs=n_jobs)(
            delayed(predict_user)(i, batch_users=batch_users, k=k)
            for i, batch_users in enumerate(inputs)
        )
        scored = sorted(scored, key=lambda x: x[0])
        scored = [s[1] for s in scored]
        return pd.concat(scored, axis=0)
