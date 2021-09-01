import unittest

import torch

from PyTorchCML.regularizers import L2Regularizer


class TestL2Regularizer(unittest.TestCase):
    """Test for Global Orthogonal Regularizer"""

    def test_forward(self):
        """
        test forward
        """
        embeddings_dict = {
            "user_embedding": torch.ones(2, 1, 5),
            "pos_item_embedding": torch.ones(2, 1, 5),
            "neg_item_embedding": torch.ones(2, 3, 5),
        }

        regularizer = L2Regularizer(weight=1)

        reg = regularizer(embeddings_dict).item()

        self.assertEqual(reg, 50)


if __name__ == "__main__":
    unittest.main()
