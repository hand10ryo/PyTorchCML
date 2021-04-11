
import pandas as pd
import torch

from ..models import BaseEmbeddingModel


class BaseEvaluator:
    def __init__(self, test_set: torch.Tensor):
        """
        Args:
            test_set: tensor of shape (n_pairs, 3) which column is [user_id, item_id, rating]
        """
        self.test_set = test_set

    def score(self, model: BaseEmbeddingModel, verbose=True) -> pd.DataFrame:

        raise NotImplementedError()
