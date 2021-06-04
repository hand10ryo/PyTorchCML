import torch
from torch import nn


class BaseRegularizer(nn.Module):
    def __init__(self, weight=1e-2):
        super().__init__()
        self.weight = weight

    def forward(self, embeddings_dict: dict) -> torch.Tensor:
        """method of forwarding

        Args:
            embeddings_dict (dict): dictionary of embbedings which will be used.

        Raises:
            NotImplementedError: [description]

        Returns:
            torch.Tensor: term of regularize
        """
        raise NotImplementedError
