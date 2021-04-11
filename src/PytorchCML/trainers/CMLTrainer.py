from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from ..evaluators import UserwiseEvaluator
from .BaseTrainer import BaseTrainer


class CMLTrainer(BaseTrainer):

    def fit(self, n_batch: int = 500, n_epoch: int = 10,
            valid_evaluator: Optional[UserwiseEvaluator] = None,
            valid_per_epoch: int = 5):

        # set evaluator and log dataframe
        valid_or_not = valid_evaluator is not None
        if valid_or_not:
            self.valid_scores = pd.DataFrame(
                valid_evaluator.score(self.model)).T
            self.valid_scores["epoch"] = 0
            self.valid_scores["loss"] = np.nan

        # start training
        for ep in range(n_epoch):
            accum_loss = 0

            # start mini-batch
            with tqdm(range(n_batch), total=n_batch) as pbar:
                for b in pbar:
                    # batch sampling
                    batch = self.sampler.get_pos_batch()
                    users = batch[:, 0:1]
                    pos_items = batch[:, 1:]

                    if self.sampler.two_stage:
                        neg_candidates = self.sampler.get_and_set_candidates()
                        dist = self.model.spreadout_distance(
                            pos_items, neg_candidates)
                        self.sampler.set_candidates_weight(dist, model.n_dim)

                    neg_items = self.sampler.get_neg_batch()

                    # initialize gradient
                    self.model.zero_grad()

                    # compute distance
                    # batch_size × 1 × 1
                    pos_dist = self.model(users, pos_items)
                    # batch_size × 1 × n_neg_samples
                    neg_dist = self.model(users, neg_items)

                    # compute loss
                    loss = self.criterion(pos_dist, neg_dist)
                    accum_loss += loss.item()

                    # gradient of loss
                    loss.backward()

                    # update model parameters
                    self.optimizer.step()

                    pbar.set_description_str(
                        f'epoch{ep+1} avg_loss:{accum_loss / (b+1) :.3f}'
                    )

            # compute metrics for epoch
            if valid_or_not and ((ep+1) % valid_per_epoch == 0) or (ep == n_epoch-1):
                valid_scores_sub = pd.DataFrame(
                    valid_evaluator.score(self.model)).T
                valid_scores_sub["epoch"] = ep + 1
                valid_scores_sub["loss"] = accum_loss / n_batch
                self.valid_scores = pd.concat(
                    [self.valid_scores, valid_scores_sub])
