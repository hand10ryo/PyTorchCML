import unittest

import torch

from PyTorchCML.regularizers import GlobalOrthogonalRegularizer


class TestGlobalOrthogonalRegularizer(unittest.TestCase):
    """Test for Global Orthogonal Regularizer"""

    def test_forward(self):
        """
        test forward
        """
        embeddings_dict = {
            "pos_item_embedding": torch.ones(2, 1, 5),
            "neg_item_embedding": torch.ones(2, 3, 5),
        }

        regularizer = GlobalOrthogonalRegularizer(weight=1)

        reg = regularizer(embeddings_dict).item()

        self.assertAlmostEqual(reg, 49.8, places=2)


if __name__ == "__main__":
    unittest.main()
