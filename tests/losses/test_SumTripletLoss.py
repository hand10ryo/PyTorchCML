import unittest

import torch
from torch import nn

from PytorchCML.losses import SumTripletLoss


class SampleRegularizer(nn.Module):
    def forward(self, embeding_dict: dict) -> torch.Tensor:
        return torch.ones(3).sum()


class TestSumTripletLoss(unittest.TestCase):
    """Test SumTripletLoss"""

    def test_forward(self):
        """
        test forward
        """
        embeddings_dict = {
            "user_embedding": torch.ones(3, 1, 5),
            "pos_item_embedding": torch.ones(3, 1, 5) * 2,
            "neg_item_embedding": torch.ones(3, 1, 5),
        }
        batch = torch.ones([3, 2])
        column_names = {"user_id": 0, "item_id": 1}

        # without regularizer
        criterion = SumTripletLoss(margin=1)
        loss = criterion(embeddings_dict, batch, column_names).item()
        self.assertGreater(loss, 0)
        self.assertEqual(loss, 6)

        # with regularizer
        regs = [SampleRegularizer()]
        criterion = SumTripletLoss(margin=1, regularizers=regs)
        loss = criterion(embeddings_dict, batch, column_names).item()
        self.assertGreater(loss, 0)
        self.assertEqual(loss, 9)


if __name__ == "__main__":
    unittest.main()
