import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import ndcg_score, average_precision_score, recall_score
from ..models import CollaborativeMetricLearning as cml


class UserwiseEvaluator:
    def __init__(self, test_set: torch.Tensor, score_function_dict: dict, ks: int = [5]):
        """
        Args:
            model: cml model
            testdata: ndarray of shape (n_pairs, 3) which column is [user_id, item_id, rating]
        """
        self.test_set = test_set
        self.score_function_dict = score_function_dict
        self.ks = ks

        self.metrics_names = [
            f"{name}@{k}"
            for k in ks
            for name in score_function_dict.keys()
        ]

    def compute_score(self, y_test_user: np.ndarray, y_hat_user: np.ndarray):

        if y_test_user.sum() == 0:
            return pd.DataFrame({name: [0] for name in self.metrics_names})

        else:
            df_eval_sub = pd.DataFrame({
                f"{name}@{k}": [metric(y_test_user, y_hat_user, k)]
                for k in self.ks
                for name, metric in self.score_function_dict.items()
            })

        return df_eval_sub

    def eval_user(self, model, uid: int):
        """user ごとに評価値を得る"""
        user_indices = (self.test_set[:, 0] == uid)
        test_set_pair = self.test_set[user_indices, :2]

        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()
        y_test_user = self.test_set[user_indices, 2].to("cpu").detach().numpy()

        return self.compute_score(y_test_user, y_hat_user)

    def score(self, model: cml, reduction="mean", verbose=True) -> pd.Series:
        """全ユーザーに対して評価値を計算して平均をとる"""

        users = torch.unique(self.test_set[:, 0])
        df_eval = pd.DataFrame({name: [] for name in self.metrics_names})

        if verbose:
            for uid in tqdm(users):
                df_eval_sub = self.eval_user(model, uid)
                df_eval = pd.concat([df_eval, df_eval_sub])
        else:
            for uid in users:
                df_eval_sub = self.eval_user(model, uid)
                df_eval = pd.concat([df_eval, df_eval_sub])

        if reduction == "mean":
            score = df_eval.mean(axis=0)

        else:
            score = df_eval.copy()

        return score


def ndcg(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):
    y_test_user = y_test_user.reshape(1, -1)
    y_hat_user = y_hat_user.reshape(1, -1)
    return ndcg_score(y_test_user, y_hat_user, k=k)


def average_precision(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):

    return average_precision_score(y_test_user, y_hat_user)


def recall(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):
    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)
    return recall_score(y_test_user, pred_topk_flag)
