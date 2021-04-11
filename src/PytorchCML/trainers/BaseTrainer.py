from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

from ..evaluators import BaseEvaluator
from ..losses import BasePairwiseLoss, BaseTripletLoss
from ..models import BaseEmbeddingModel
from ..samplers import BaseSampler


class BaseTrainer:
    def __init__(self,
                 model: BaseEmbeddingModel,
                 optimizer: optim,
                 criterion: Union[BasePairwiseLoss, BaseTripletLoss],
                 sampler: BaseSampler):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.sampler = sampler

    def fit(self, n_batch: int = 500, n_epoch: int = 10,
            valid_evaluator: Optional[BaseEvaluator] = None,
            valid_per_epoch: int = 5):

        raise NotImplementedError
