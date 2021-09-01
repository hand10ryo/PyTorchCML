import torch

from .BaseEmbeddingModel import BaseEmbeddingModel


class CollaborativeMetricLearning(BaseEmbeddingModel):
    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> dict:
        """
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
        }

        return embeddings_dict

    def spreadout_distance(self, pos_items: torch.Tensor, neg_itmes: torch.Tensor):
        """
        Args:
           pos_items : tensor of user indices size (n_batch, 1).
           neg_itmes : tensor of item indices size (n_neg_candidates)
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
            dist : distance for each users and item pair size (n_pairs)
        """
        # set users and user
        users = pairs[:, :1]
        items = pairs[:, 1:2]

        # get enmbeddigs
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)

        # compute distance
        dist = torch.cdist(u_emb, i_emb).reshape(-1)

        max_dist = 2 * self.max_norm if self.max_norm is not None else 100

        return max_dist - dist
