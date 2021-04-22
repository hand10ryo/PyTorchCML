import unittest

import torch
import numpy as np

from PytorchCML.losses import LogitPairwiseLoss


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


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
        criterion = LogitPairwiseLoss()
        pos_inner_data = torch.ones(2, 1) * 3
        neg_inner_data = - torch.ones(2, 3) * 3
        loss = criterion(pos_inner_data, neg_inner_data).item()

        pos_loss = - np.log(sigmoid(
            pos_inner_data.to("cpu").detach().numpy()
        )).sum()
        neg_loss = - np.log(sigmoid(
            - neg_inner_data.to("cpu").detach().numpy()
        )).sum()
        loss_np = (pos_loss + neg_loss) / (2 * (1 + 3))
        self.assertAlmostEqual(loss, loss_np)


if __name__ == '__main__':
    unittest.main()
