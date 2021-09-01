import torch
from torch import nn


class BasePairwiseLoss(nn.Module):
    """Class of abstract loss module for pairwise loss like matrix factorization."""

    def __init__(self, regularizers: list = []):
        super().__init__()
        self.regularizers = regularizers

    def forward(
        self, embeddings_dict: dict, batch: torch.Tensor, column_names: dict
    ) -> torch.Tensor:
        """
        Args:
            embeddings_dict (dict): A dictionary of embddings which has following key and values.
            batch (torch.Tensor) : A tensor of batch, size (n_batch, *).
            column_names (dict) : A dictionary that maps names to indices of rows of batch.

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: [description]

         ---   example code   ---

        # embeddings_dict = {
        #   "user_embedding": user_emb,
        #    "pos_item_embedding": pos_item_emb,
        #    "neg_item_embedding": neg_item_emb,
        #    "user_bias": user_bias,
        #    "pos_item_bias": pos_item_bias,
        #    "neg_item_bias": neg_item_bias,
        #}

        loss = loss_function(embeddings_dict, batch, column_names)
        reg = self.regularize(embeddings_dict)
        return loss + reg
        """

        raise NotImplementedError

    def regularize(self, embeddings_dict: dict):
        reg = 0
        for regularizer in self.regularizers:
            reg += regularizer(embeddings_dict)

        return reg
