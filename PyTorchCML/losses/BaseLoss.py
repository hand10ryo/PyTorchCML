import torch
from torch import nn


class BaseLoss(nn.Module):
    """Class of abstract loss module for pairwise loss like matrix factorization."""

    def __init__(self, regularizers: list = []):
        super().__init__()
        self.regularizers = regularizers

    def forward(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        loss = self.main(embeddings_dict, batch, column_names)
        loss += self.regularize(embeddings_dict)
        return loss

    def main(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        """
        Args:
            embeddings_dict (dict): A dictionary of embddings.
            (e.g. It has following key and values.)
                user_embedding : embeddings of user, size (n_batch, 1, d)
                pos_item_embedding : embeddings of positive item, size (n_batch, 1, d)
                neg_item_embedding : embeddings of negative item, size (n_batch, n_neg_samples, d)
                user_bias : bias of user, size (n_batch, 1)
                pos_item_bias : bias of positive item, size (n_batch, 1)
                neg_item_bias : bias of negative item, size (n_batch, n_neg_samples)

            batch (torch.Tensor) : A tensor of batch, size (n_batch, *).
            column_names (dict) : A dictionary that maps names to indices of rows of batch.

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [description]

         ---   example code   ---

        embeddings_dict = {
           "user_embedding": user_embedding,
            "pos_item_embedding": pos_item_embedding,
            "neg_item_embedding": neg_item_embedding,
            "user_bias": user_bias,
            "pos_item_bias": pos_item_bias,
            "neg_item_bias": neg_item_bias,
        }

        loss = loss_function(embeddings_dict, batch, column_names)

        return loss
        """

        raise NotImplementedError

    def regularize(self, embeddings_dict: dict):
        reg = 0
        for regularizer in self.regularizers:
            reg += regularizer(embeddings_dict)

        return reg
