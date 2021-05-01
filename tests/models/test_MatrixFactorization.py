import unittest

import torch

from PytorchCML.models import LogitMatrixFactorization


class TestLogitMatrixFactorization(unittest.TestCase):
    def test_forward(self):
        users = torch.LongTensor([0, 1])
        pos_items = torch.LongTensor([2, 3])
        neg_items = torch.LongTensor([[1, 3, 4], [0, 1, 2]])

        model = LogitMatrixFactorization(
            n_user=3,
            n_item=5,
            n_dim=10,
        )

        u_emb, ip_emb, in_emb, ub, ipb, inb = model(users, pos_items, neg_items)

        # user_emb shape
        shape = u_emb.shape
        self.assertEqual(shape, (2, 10))

        # pos_item_emb shape
        shape = ip_emb.shape
        self.assertEqual(shape, (2, 10))

        # neg_item_emb shape
        shape = in_emb.shape
        self.assertEqual(shape, (2, 3, 10))

        # user_bias shape
        shape = ub.shape
        self.assertEqual(shape, (2, 1))

        # pos_item_bias shape
        shape = ipb.shape
        self.assertEqual(shape, (2, 1))

        # neg_item_bias shape
        shape = inb.shape
        self.assertEqual(shape, (2, 3, 1))

    def test_predict(self):
        user_item_pairs = torch.LongTensor([[0, 0], [1, 1], [2, 2]])
        model = LogitMatrixFactorization(
            n_user=3,
            n_item=5,
            n_dim=10,
        )

        y_hat = model.predict(user_item_pairs)

        # y_hat shape
        shape = y_hat.shape
        self.assertEqual(shape, torch.Size([3]))

    def test_predict_proba(self):
        user_item_pairs = torch.LongTensor([[0, 0], [1, 1], [2, 2]])
        model = LogitMatrixFactorization(
            n_user=3,
            n_item=5,
            n_dim=10,
        )

        y_hat = model.predict_proba(user_item_pairs)

        # y_hat shape
        shape = y_hat.shape
        self.assertEqual(shape, torch.Size([3]))

        # range
        min_y_hat = y_hat.min().item()
        max_y_hat = y_hat.max().item()
        self.assertGreater(min_y_hat, 0)
        self.assertGreater(1, max_y_hat)

    def test_get_item_weight(self):
        users = torch.LongTensor([0, 1])
        model = LogitMatrixFactorization(
            n_user=3,
            n_item=5,
            n_dim=10,
        )
        item_weight = model.get_item_weight(users)

        # y_hat shape
        shape = item_weight.shape
        self.assertEqual(shape, (2, 5))

        # range
        min_weight = item_weight.min().item()
        self.assertGreater(min_weight, 0)

        # value test
        model = LogitMatrixFactorization(
            n_user=3,
            n_item=5,
            n_dim=10,
            user_embedding_init=torch.ones(3, 10),
            item_embedding_init=torch.ones(5, 10),
            user_bias_init=torch.ones(3),
            item_bias_init=torch.ones(5),
        )
        model.weight_link = lambda x: x
        item_weight = model.get_item_weight(users)

        min_weight = item_weight.min().item()
        max_weight = item_weight.max().item()
        self.assertEqual(min_weight, 12)
        self.assertEqual(max_weight, 12)
