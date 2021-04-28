import unittest
import torch
from torch import optim

from PytorchCML import evaluators, losses, models, samplers, trainers


class TestMFTrainer(unittest.TestCase):
    def test_fit(self):
        train_set = torch.LongTensor([[0, 0], [0, 2], [1, 1], [1, 3], [2, 3], [2, 4]])
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

        lr = 1e-3
        n_user, n_item, n_dim = 3, 5, 10
        model = models.LogitMatrixFactorization(n_user, n_item, n_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = losses.LogitPairwiseLoss()
        sampler = samplers.BaseSampler(train_set, n_user, n_item, n_neg_samples=2)

        score_function_dict = {
            "nDCG": evaluators.ndcg,
            "MAP": evaluators.average_precision,
            "Recall": evaluators.recall,
        }

        evaluator = evaluators.UserwiseEvaluator(
            test_set, score_function_dict, ks=[2, 3]
        )

        trainer = trainers.MFTrainer(model, optimizer, criterion, sampler)

        trainer.fit(n_batch=3, n_epoch=3, valid_evaluator=evaluator, valid_per_epoch=1)

        df_eval = trainer.valid_scores

        self.assertEqual(df_eval.shape, (4, 8))


if __name__ == "__main__":
    unittest.main()
