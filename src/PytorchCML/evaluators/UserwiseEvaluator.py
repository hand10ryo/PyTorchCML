import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, ndcg_score, recall_score
from tqdm import tqdm

from ..models import BaseEmbeddingModel
from .BaseEvaluator import BaseEvaluator


class UserwiseEvaluator(BaseEvaluator):
    """Class of evaluator computing metrics for each user and calcurating average."""

    def __init__(
        self, test_set: torch.Tensor, score_function_dict: dict, ks: list = [5]
    ):
        """Set test data and metrics.

        Args:
            test_set (torch.Tensor): test data which column is [user_id, item_id, rating].
            score_function_dict (dict): dictionary whose keys are metrics name and values are user-wise function.
            ks (int, optional): A list of @k. Defaults to [5].

        for example, score_function_dict is

        score_function_dict = {
            "nDCG" : evaluators.ndcg,
            "MAP" : evaluators.average_precision,
            "Recall": evaluators.recall
        }

        arguments of each functions must be
            y_test_user (np.ndarray): grand truth for the user
            y_hat_user (np.ndarray) : prediction of relevance
            k : a number of top item considered.
        """
        super().__init__(test_set)

        self.score_function_dict = score_function_dict
        self.ks = ks

        self.metrics_names = [
            f"{name}@{k}" for k in ks for name in score_function_dict.keys()
        ]

    def compute_score(
        self, y_test_user: np.ndarray, y_hat_user: np.ndarray
    ) -> pd.DataFrame:
        """Method of computing score.
        This method make a row of DataFrame which has scores for each metrics and k for the user.

        Args:
            y_test_user (np.ndarray): [description]
            y_hat_user (np.ndarray): [description]

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """

        if y_test_user.sum() == 0:
            return pd.DataFrame({name: [0] for name in self.metrics_names})

        else:
            df_eval_sub = pd.DataFrame(
                {
                    f"{name}@{k}": [metric(y_test_user, y_hat_user, k)]
                    for k in self.ks
                    for name, metric in self.score_function_dict.items()
                }
            )

        return df_eval_sub

    def eval_user(self, model: BaseEmbeddingModel, uid: int) -> pd.DataFrame:
        """Method of evaluating for given user.

        Args:
            model (BaseEmbeddingModel): model which have user and item embeddings.
            uid (int): user id

        Returns:
            (pd.DataFrame): a row of DataFrame which has scores for each metrics and k for the user.
        """
        user_indices = self.test_set[:, 0] == uid
        test_set_pair = self.test_set[user_indices, :2]

        y_hat_user = model.predict(test_set_pair).to("cpu").detach().numpy()
        y_test_user = self.test_set[user_indices, 2].to("cpu").detach().numpy()

        return self.compute_score(y_test_user, y_hat_user)

    def score(
        self, model: BaseEmbeddingModel, reduction="mean", verbose=True
    ) -> pd.DataFrame:
        """Method of calculating average score for all users.

        Args:
            model (BaseEmbeddingModel): model which have user and item embeddings.
            reduction (str, optional): reduction method. Defaults to "mean".
            verbose (bool, optional): displaying progress bar or not during evaluating. Defaults to True.

        Returns:
            pd.DataFrame: a row of DataFrame which has average scores
        """

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
            score = pd.DataFrame(df_eval.mean(axis=0)).T

        else:
            score = df_eval.copy()

        return score


def ndcg(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int) -> float:
    """Function for user-wise evaluator calculating ndcg @ k

    Args:
        y_test_user (np.ndarray): grand truth for the user
        y_hat_user (np.ndarray): prediction of relevance
        k (int): a number of top item considered.

    Returns:
        (float): ndcg score
    """
    y_test_user = y_test_user.reshape(1, -1)
    y_hat_user = y_hat_user.reshape(1, -1)
    return ndcg_score(y_test_user, y_hat_user, k=k)


def average_precision(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):
    """Function for user-wise evaluator calculating average precision (MAP) @ k

    Args:
        y_test_user (np.ndarray): grand truth for the user
        y_hat_user (np.ndarray): prediction of relevance
        k (int): a number of top item considered.

    Returns:
        (float): average precision score
    """
    pred_sort_indices = (-y_hat_user).argsort()
    topk_y_hat = y_hat_user[pred_sort_indices[:k]]
    topk_y_test = y_test_user[pred_sort_indices[:k]]

    if topk_y_test.sum() < 1:
        return 0
    else:
        return average_precision_score(topk_y_test, topk_y_hat)


def recall(y_test_user: np.ndarray, y_hat_user: np.ndarray, k: int):
    """Function for user-wise evaluator calculating Recall @ k

    Args:
        y_test_user (np.ndarray): grand truth for the user
        y_hat_user (np.ndarray): prediction of relevance
        k (int): a number of top item considered.

    Returns:
        (float): recall score
    """
    pred_rank = (-y_hat_user).argsort().argsort() + 1
    pred_topk_flag = (pred_rank <= k).astype(int)
    return recall_score(y_test_user, pred_topk_flag)
