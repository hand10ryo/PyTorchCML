import unittest

import torch
import numpy as np

from PyTorchCML.evaluators import UserwiseEvaluator, ndcg, average_precision, recall
from PyTorchCML.models import CollaborativeMetricLearning


class TestScoreFunctions(unittest.TestCase):
    """Test Score Functions"""

    def test_ndcg(self):
        """Test ndcg

        true relevance : [0, 1, 1, 0, 1]
        predict relevance : [4, 6, 2, 0, 1]

        discount : [1/log_2(2), 1/log_2(3), 1/log_2(4), 1/log_2(5), 1/log_2(6)]
                    = [1, 0.6309, 0.5, 0.4306, 0.3868]
        true relevance ranked by predict relevance : [1, 0, 1, 1, 0]

        ndcg@2 = ( 1 + 0 ) / ( 1 + 0.6309 ) = 0.6131
        ndcg@3 = ( 1 + 0 + 0.5 ) / ( 1 + 0.631 + 0.5 ) = 0.7038
        ndcg@4 = ( 1 + 0 + 0.5 + 0.4306 ) / ( 1 + 0.6309 + 0.5 + 0) = 0.9060
        """
        y_test_user = np.array([0, 1, 1, 0, 1])
        y_hat_user = np.array([4, 6, 2, 0, 1])

        ndcg_at_2 = ndcg(y_test_user, y_hat_user, 2)
        ndcg_at_3 = ndcg(y_test_user, y_hat_user, 3)
        ndcg_at_4 = ndcg(y_test_user, y_hat_user, 4)

        self.assertAlmostEqual(ndcg_at_2, 0.6131, places=3)
        self.assertAlmostEqual(ndcg_at_3, 0.7038, places=3)
        self.assertAlmostEqual(ndcg_at_4, 0.9060, places=3)

    def test_average_precision(self):
        """Test average_precision

        true relevance : [0, 1, 1, 0, 1]
        predict relevance : [4, 6, 2, 0, 1]
        true relevance ranked by predict relevance : [1, 0, 1, 1, 0]

        AP@2 = ( 1 + 0 ) / 1 = 1
        AP@3 = ( 1 + 2/3 ) / 2 = 0.8333
        AP@4 = ( 1 + 2/3 + 3/4 ) / 3 = 0.8055
        """

        y_test_user = np.array([0, 1, 1, 0, 1])
        y_hat_user = np.array([4, 6, 2, 0, 1])

        ap_at_2 = average_precision(y_test_user, y_hat_user, 2)
        ap_at_3 = average_precision(y_test_user, y_hat_user, 3)
        ap_at_4 = average_precision(y_test_user, y_hat_user, 4)

        self.assertEqual(ap_at_2, 1)
        self.assertAlmostEqual(ap_at_3, 0.8333, places=3)
        self.assertAlmostEqual(ap_at_4, 0.8055, places=3)

    def test_recall(self):
        """Test recall

        true relevance : [0, 1, 1, 0, 1]
        predict relevance : [4, 6, 2, 0, 1]
        true relevance ranked by predict relevance : [1, 0, 1, 1, 0]

        recall@2 = 1 / 3 = 0.3333
        recall@3 = 2 / 3 = 0.6666
        recall@4 = 3 / 3 = 1
        """

        y_test_user = np.array([0, 1, 1, 0, 1])
        y_hat_user = np.array([4, 6, 2, 0, 1])

        recall_at_2 = recall(y_test_user, y_hat_user, 2)
        recall_at_3 = recall(y_test_user, y_hat_user, 3)
        recall_at_4 = recall(y_test_user, y_hat_user, 4)

        self.assertAlmostEqual(recall_at_2, 0.3333, places=3)
        self.assertAlmostEqual(recall_at_3, 0.6666, places=3)
        self.assertEqual(recall_at_4, 1)


class TestUserwiseEvaluator(unittest.TestCase):
    """Test UserwiseEvaluator"""

    def test_eval_user(self):
        score_function_dict = {"nDCG": ndcg, "MAP": average_precision, "Recall": recall}
        test_set = torch.LongTensor(
            [
                [0, 1, 1],
                [0, 3, 1],
                [0, 4, 0],
                [1, 0, 0],
                [1, 2, 1],
                [1, 4, 1],
                [2, 0, 1],
                [2, 1, 0],
                [2, 2, 1],
            ]
        )

        evaluator = UserwiseEvaluator(test_set, score_function_dict, ks=[2, 3])
        model = CollaborativeMetricLearning(n_user=3, n_item=5, n_dim=10)
        df_eval_sub = evaluator.eval_user(model, 0)

        # shape
        n, m = df_eval_sub.shape
        self.assertEqual(n, 1)
        self.assertEqual(m, 6)

        # columns
        columns = list(df_eval_sub.columns)
        self.assertEqual(
            columns, ["nDCG@2", "MAP@2", "Recall@2", "nDCG@3", "MAP@3", "Recall@3"]
        )

    def test_score(self):
        score_function_dict = {"nDCG": ndcg, "MAP": average_precision, "Recall": recall}
        test_set = torch.LongTensor(
            [
                [0, 1, 0],
                [0, 3, 1],
                [0, 4, 0],
                [1, 0, 0],
                [1, 2, 0],
                [1, 4, 1],
                [2, 0, 1],
                [2, 1, 0],
                [2, 2, 0],
            ]
        )

        evaluator = UserwiseEvaluator(test_set, score_function_dict, ks=[2, 3])
        model = CollaborativeMetricLearning(n_user=3, n_item=5, n_dim=10)

        df_eval = evaluator.score(model)

        # shape
        n, m = df_eval.shape
        self.assertEqual(n, 1)
        self.assertEqual(m, 6)

        # columns
        columns = list(df_eval.columns)
        self.assertEqual(
            columns, ["nDCG@2", "MAP@2", "Recall@2", "nDCG@3", "MAP@3", "Recall@3"]
        )

        # is not null
        for col in columns:
            self.assertIsNotNone(df_eval[col].values[0])

        # reduction
        df_eval = evaluator.score(model, reduction="sum")

        # shape
        n, m = df_eval.shape
        self.assertEqual(n, 3)
        self.assertEqual(m, 6)


if __name__ == "__main__":
    unittest.main()
