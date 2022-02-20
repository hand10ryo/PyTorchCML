# PyTorchCML

![https://github.com/hand10ryo/PyTorchCML/blob/image/images/icon.png](https://github.com/hand10ryo/PyTorchCML/blob/image/images/icon.png)

PyTorchCML is a library of PyTorch implementations of matrix factorization (MF) and collaborative metric learning (CML), algorithms used in recommendation systems and data mining.

日本語版READMEは[こちら](https://github.com/hand10ryo/PyTorchCML/blob/main/README_ja.md)

# What is CML ?

CML is an algorithm that combines metric learning and MF. It allows us to embed elements of two sets, such as user-item or document-word, into a joint distance metric space using their relational data.

In particular, CML is known to capture user-user and item-item relationships more precisely than MF and can achieve higher accuracy and interpretability than MF for recommendation systems [1]. In addition, the embeddings can be used for secondary purposes such as friend recommendations on SNS and similar item recommendations on e-commerce sites.

For more details, please refer to this reference [1].

# Installation

You can install PyTorchCML using Python's package manager pip.

```bash
pip install PyTorchCML
```

You can also download the source code directly and build your environment with poetry.。

```bash
git clone https://github.com/hand10ryo/PyTorchCML
poetory install 
```

## dependencies

The dependencies are as follows

- python = ">=3.7.10,<3.9"
- torch = "^1.8.1"
- scikit-learn = "^0.22.2"
- scipy = "^1.4.1"
- numpy = "^1.19.5"
- pandas = "^1.1.5"
- tqdm = "^4.41.1"

# Usage

## Example

[This](https://github.com/hand10ryo/PytorchCML/tree/main/examples/notebooks) is a jupyter notebook example using the Movielens 100k dataset.

## Overview

This library consists of the following six modules.

- trainers
- models
- samplers
- losses
- regularizers
- evaluators

By combining these modules, you can implement a variety of algorithms.

The following figure shows the relationship between these modules.

![https://github.com/hand10ryo/PyTorchCML/blob/image/images/diagram.png](https://github.com/hand10ryo/PyTorchCML/blob/image/images/diagram.png)

The most straightforward implementation is as follows.

```python
import torch
from torch import optim
import numpy as np
from PyTorchCML import losses, models, samplers, trainers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# train dataset (whose columns are [user_id, item_id].)
train_set = np.array([[0, 0], [0, 1], [1, 2], [1, 3]]) 
train_set_torch = torch.LongTensor(train_set).to(device)
n_user = train_set[:,0].max() + 1
n_item = train_set[:,1].max() + 1

# model settings
model = models.CollaborativeMetricLearning(n_user, n_item, n_dim=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = losses.MinTripletLoss(margin=1).to(device)
sampler = samplers.BaseSampler(train_set_torch, n_user, n_item, device=device)
trainer = trainers.BaseTrainer(model, optimizer, criterion, sampler)

# run 
trainer.fit(n_batch=256, n_epoch=3)
```

The input `train_set` represents a two-column NumPy array whose records are the user IDs and item IDs that received positive feedback. 

The `n_user` and `n_item` are the number of users and items. Here, we assume that user ID and item ID start from 0 and that all users and items are included in the train_set.

Then, define model, optimizer, criterion, and sampler, input them to a trainer and run the trainer's fit method to start learning CM

## models

The models is the module that handles the embeddings.

There are currently two models to choose from as follows.

- models.CollaborativeMetricLearning
- models.LogitMatrixFactorization

You can predict the relationship between the target user and the item with the `predict` method.

CML uses vector distance, while MF uses the inner product to represent the relationship.

You can also set the maximum norm and initial value of the embeddings.

For example, in `LogitMatrixFactorization`, this is how it works.

```python
model = models.LogitMatrixFactorization(
    n_user, n_item, n_dim, max_norm=5,
    user_embedding_init = torch.Tensor(U),   # shape = (n_user, n_dim)
    item_embedding_init = torch.Tensor(V.T), # shape = (n_dim, n_item)
).to(device)
```

## losses

The losses module is for handling the loss function for learning embeddings.
We can mainly divide the loss function into PairwiseLoss and TripletLoss.

PairwiseLoss is the loss for each user-item pair <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;(u,i)" title="(u, i)" />.

TripletLoss is the loss per <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;(u,i_+,i_-)" title="(u,i_+,i_-)" />.
Here, <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;(u,i_+)" title="(u,i_+)" /> is a positive pair, and <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;(u,i_-)" title="(u,i_-)" /> is a negative pair.

In general, CML uses triplet loss, and MF uses pairwise loss.

## samplers

The samplers is a module that handles the sampling of mini-batches during training.

There are two types of sampling done by the sampler.

- Sampling of positive user-item pairs <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;(u,i_+)" title="(u,i_+)" />
- Sampling of negative items <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;i_-" title="i_-" />

The default setting is to sample both with a uniform random probability.

It is also possible to weigh both positively and negatively.

For example, if you want to weigh the items by their popularity, you can follow.

```python
item_ids, item_popularity = np.unique(train_set[:,1], return_counts=True)
sampler = samplers.BaseSampler(
    train_set_torch, neg_weight = item_popularity,
    n_user, n_item, device=device
)
```

## trainers

The trainers is the module that handles training.

You can train by setting up a model, optimizer, loss function, and sampler.

## evaluators

The evaluators is a module for evaluating performance after learning.

You can evaluate your model as follows.

```python
from PyTorchCML import evaluators

# test set (whose columns are [user_id, item_id, rating].)
test_set = np.array([[0, 2, 3], [0, 3, 4], [1, 0, 2], [1, 1, 5]])
test_set_torch = torch.LongTensor(test_set).to(device)

# define metrics and evaluator
score_function_dict = {
    "nDCG" : evaluators.ndcg,
    "MAP" : evaluators.average_precision,
    "Recall": evaluators.recall
}
evaluator = evaluators.UserwiseEvaluator(
    test_set_torch, 
    score_function_dict, 
    ks=[3,5]
)

# calc scores
scores = evaluator.score(model)
```

The `test_set` is a three-column NumPy array with user ID, item ID, and rating records.

The `score_function_dict` is a dictionary of evaluation metrics. Its key is a name, and its value is a function to compute the evaluation metric. The evaluators module implements nDCG@k, MAP@k, and Recall@k as its functions. In this example, those three are set, but you can set any number of evaluation indicators.

The `evaluator` takes input test data, evaluation metrics, and a list with @k types. 

You can calculate the scores by running the method `.score()` with the model as input.  Its output `scores` will be a single row pandas.DataFrame with each score. In this example, its columns are `["nDCG@3", "MAP@3", "Recall@3", "nDCG@5", "MAP@5", "Recall@5"]`.

Also, inputting the evaluator to the `valid_evaluator` argument of the fit method of the trainer will allow you to evaluate the learning progress.
This system is helpful for hyperparameter tuning.

```python
valid_evaluator = evaluators.UserwiseEvaluator(
    test_set_torch, # eval set
    score_function_dict, 
    ks=[3,5]
)
trainer.fit(n_batch=50, n_epoch=15, valid_evaluator = valid_evaluator)
```

## regularizers

The regularizers is a module that handles the regularization terms of embedded vectors.

You can implement the L2 norm, etc., by entering a list of regularizer instances as the argument of the loss function, as shown below.

```python
from PyTorchCML import regularizers
regs = [regularizers.L2Regularizer(weight=1e-2)]
criterion = losses.MinTripletLoss(margin=1, regularizers=regs).to(device)
```

It is also possible to introduce multiple regularizations by increasing the length of the list.

## adaptors

The adaptors is a module for realizing domain adaptation.

Domain adaptation in CML is achieved by adding <img src="https://latex.codecogs.com/gif.latex?\inline&space;\bg_black&space;L(v_i,&space;\theta)&space;=&space;\|f(x_i;\theta)-v_i\|^2" title="L(v_i, \theta) = \|f(x_i;\theta)-v_i\|^2" /> to the loss for feature <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;x_i" title="x_i" /> of item  <img src="https://latex.codecogs.com/gif.latex?\bg_black&space;i" title="i" /> . The same is true for the user. This allows us to reflect attribute information in the embedding vector.

MLPAdaptor is a class of adaptors that assumes a multilayer perceptron in function <img src="https://latex.codecogs.com/gif.latex?\inline&space;\bg_black&space;f(x_i;\theta)" title="f(x_i;\theta)" />.

You can set up the adaptor as shown in the code below

```python
from PyTorchCML import adaptors

# item_feature.shape = (n_item, n_feature)
item_feature_torch = torch.Tensor(item_feature)
adaptor = adaptors.MLPAdaptor(
    item_feature_torch, 
    n_dim=10, 
    n_hidden=[20], 
    weight=1e-4
)

model = models.CollaborativeMetricLearning(
    n_user, n_item, n_dim, 
    item_adaptor=adaptor
).to(device)
```

# Development

```bash
pip install poetry
pip install poetry-dynamic-versioning

poetry install
poetry build
poetry lock
```

# Citation

You may use PyTorchCML under MIT License. If you use this program in your research then please cite:

```jsx
@misc{matsui2021pytorchcml,
  author = {Ryo, Matsui},
  title = {PyTorchCML},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/hand10ryo/PyTorchCML}
}
```

# References

[1] Cheng-Kang Hsieh, Longqi Yang, Yin Cui, Tsung-Yi Lin, Serge Belongie, and Deborah Estrin.Collaborative metric learning. InProceedings of the 26th International Conference on World WideWeb, pp. 193–201, 2017.