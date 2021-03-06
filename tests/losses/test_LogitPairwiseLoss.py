import unittest

import torch
from torch import nn
import numpy as np

from PyTorchCML.losses import LogitPairwiseLoss


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class SampleRegularizer(nn.Module):
    def forward(self, embeding_dict: dict) -> torch.Tensor:
        return torch.ones(3).sum()


class TestLogitPairwiseLoss(unittest.TestCase):
    """Test LogitPairwiseLoss"""

    def test_forward(self):
        """
        test forward

        pos_dist = [[3], [3]]
        neg_dist = [[0,1,2], [3,4,5]]
        loss = [[10], [1]]
        avg_loss = 5.5
        """
        embeddings_dict = {
            "user_embedding": torch.ones(3, 1, 5),
            "pos_item_embedding": torch.ones(3, 1, 5) * 2,
            "neg_item_embedding": torch.ones(3, 2, 5),
            "user_bias": torch.zeros(3, 1),
            "pos_item_bias": torch.zeros(3, 1) * 2,
            "neg_item_bias": torch.zeros(3, 2),
        }
        batch = torch.ones([3, 2])
        column_names = {"user_id": 0, "item_id": 1}

        # without regularizer
        criterion = LogitPairwiseLoss()
        loss = criterion(embeddings_dict, batch, column_names).item()

        self.assertGreater(loss, 0)
        self.assertAlmostEqual(loss, 3.3378, places=3)

        # with regularizer
        regs = [SampleRegularizer()]
        criterion = LogitPairwiseLoss(regularizers=regs)
        loss = criterion(embeddings_dict, batch, column_names).item()
        self.assertGreater(loss, 0)
        self.assertAlmostEqual(loss, 6.3378, places=3)


if __name__ == "__main__":
    unittest.main()
