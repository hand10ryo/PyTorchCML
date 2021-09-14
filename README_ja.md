# PyTorchCML

![https://github.com/hand10ryo/PyTorchCML/blob/image/images/icon.png](https://github.com/hand10ryo/PyTorchCML/blob/image/images/icon.png)

PyTorchCMLは、推薦システム・データマイニングのアルゴリズムである 行列分解(matrix factorization, MF) および collaborative metric learning (CML)を PyTorch で実装したライブラリです。

English version of README is [here](https://github.com/hand10ryo/PyTorchCML/blob/main/README.md)

# CML とは

CML は metric learning と MF を組み合わせたアルゴリズムで、ユーザー×アイテム、ドキュメント × 単語 など 2つの集合の要素をそれらの関係データを用いて同じ距離空間に埋め込むことを可能にします。

特に、CML は MF よりもユーザー間およびアイテム間の関係性を精緻に捉えられることがわかっています[1] 。そのため、推薦システムにおいて MF よりも高い精度が望まれる上、解釈性が高く定性的な評価が容易となると考えられています。また、SNS上の友達推薦やECサイト上の類似商品推薦など埋め込みベクトルの副次的な利用も想定できます。

詳しくはこちらの参考文献 [1] をご参照ください。

# インストール

PytorchCMLは python のパッケージマネージャー pip でインストール可能です。

```bash
pip install PyTorchCML
```

また、ソースコード を直接 ダウンロードして poetry で環境を構築することもできます。

```bash
git clone https://github.com/hand10ryo/PyTorchCML
poetory install 
```

## 依存関係

依存ライブラリは以下の通りです。

- python = ">=3.7.10,<3.9"
- torch = "^1.8.1"
- scikit-learn = "^0.22.2"
- scipy = "^1.4.1"
- numpy = "^1.19.5"
- pandas = "^1.1.5"
- tqdm = "^4.41.1"

# 使い方

## 例

Movielens 100k データセットを用いた jupyter notebook の例が[こちら](https://github.com/hand10ryo/PytorchCML/tree/main/examples/notebooks)にあります。

## 概観

このライブラリは以下の6つのモジュールで構成されています。

- trainers
- models
- samplers
- losses
- regularizers
- evaluators

これらを組み合わせることで様々なアルゴリズムを実装可能となります。

これらのモジュールは以下の図のような関係があります。

![https://github.com/hand10ryo/PyTorchCML/blob/image/images/diagram.png](https://github.com/hand10ryo/PyTorchCML/blob/image/images/diagram.png)

最も単純化した実装は以下の通りです。

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

ただし `train_set`は、ポジティブなfeedback を受け取ったユーザーIDおよびアイテムIDをレコードにもつ２列の numpy 配列を表します。

また`n_user`および`n_item`はユーザー数・アイテム数の取得をしています。ユーザーIDおよびアイテムIDは0から始まり、全てのユーザーおよびアイテムが`train_set`に含まれることを想定しています。

その後、model, optimizer, criterion, sampler を定義し、trainer に入力し、trainer の fit メソッドを実行すると CML の学習が始まります。

## models

models 埋め込みベクトルを司るモジュールです。

モデルは現在、以下の二つが選べます。

- models.CollaborativeMetricLearning
- models.LogitMatrixFactorization

`predict` メソッドで対象のユーザーとアイテムの関係性を予測できます。

CMFはベクトル距離、MFの内積で関係性を表現します。

また、ベクトルの最大ノルム、埋め込みベクトルの初期値を設定することもできます。

例えば LogitMatrixFactorizationではこのようになります。

```python
model = models.LogitMatrixFactorization(
    n_user, n_item, n_dim, max_norm=5,
    user_embedding_init = torch.Tensor(U),   # shape = (n_user, n_dim)
    item_embedding_init = torch.Tensor(V.T), # shape = (n_dim, n_item)
).to(device)
```

## losses

losses は埋め込みベクトル学習のための損失関数を司るモジュールです。

損失関数は主に、PairwiseLoss と TripletLoss に分けられます。

PairwiseLoss は、ユーザーアイテムペア <img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;(u,i)" title="(u, i)" /> ごとの損失です。

TripletLoss は、ポジティブなユーザーアイテムペア <img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;(u,i_+)" title="(u,i_+)" />に対してネガティブなアイテム<img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;i_-" title="i_-" />を加えた<img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;(u,i_+,i_-)" title="(u,i_+,i_-)" />ごとの損失です。

## samplers

samplers は学習中のミニバッチのサンプリングを司るモジュールです。

sampler が行うサンプリングは２種類あります。

- ポジティブなユーザーアイテムペア<img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;(u,i_+)" title="(u,i_+)" />の抽出
- ネガティブなアイテム<img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;i_-" title="i_-" />の抽出

デフォルトでは両者のサンプリングを一様ランダムに行います。

ポジティブおよびネガティブで共に重み付けすることも可能です。

例えば、アイテムの人気度で重み付けする場合は以下のように行います。

```python
item_ids, item_popularity = np.unique(train_set[:,1], return_counts=True)
sampler = samplers.BaseSampler(
    train_set_torch, neg_weight = item_popularity,
    n_user, n_item, device=device
)
```

## trainers

trainers は学習を司るモジュールです。

モデル、オプティマイザ、損失関数、サンプラーを設定すると学習ができます。

## evaluators

evaluators は学習後のパフォーマンス評価を行うためのモジュールです。

学習後の評価は以下のように行うことができます。

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

1行目の入力 `test_set`は、評価対象ユーザーアイテムペアのID およびその評価値をレコードにもつ3列の numpy 配列を表します。

`score_function_dict`は評価指標の定義です。

key にその名前、value には評価指標を計算する関数を設定します。

evaluators モジュールにはその関数として、nDCG@k, MAP@k, Recall@k が実装されています。

ここではその３つを設定していますが、任意の数の評価指標を設定できます。

`evaluator`は、テストデータ、評価指標、 @k の種類を持つリストを入力とします。model を入力とするメソッド `.score()`を実行すればそのスコアを計算できます。その出力 `scores`は、各スコアを持つ1行の pandas.DataFrame となります。この例ではそのカラムは `["nDCG@3", "MAP@3", "Recall@3", "nDCG@5", "MAP@5", "Recall@5"]`となります。

また、trainer の fit メソッドの引数 `valid_evaluator` に evaluator を設定すれば学習経過にも評価することができ、ハイパーパラメータ調整にも役立ちます。

```python
valid_evaluator = evaluators.UserwiseEvaluator(
    test_set_torch, # eval set
    score_function_dict, 
    ks=[3,5]
)
trainer.fit(n_batch=50, n_epoch=15, valid_evaluator = valid_evaluator)
```

## regularizers

regularizers は埋め込みベクトルの正則化項を司るモジュールです。

以下のように、損失関数の引数に regularizer のインスタンスを要素にもつリストを入力することでL2ノルムなどを実装することができます。

```python
from PyTorchCML import regularizers
regs = [regularizers.L2Regularizer(weight=1e-2)]
criterion = losses.MinTripletLoss(margin=1, regularizers=regs).to(device)
```

リストの長さを増やせば複数の正則化を導入することも可能です。

## adaptors

adaptors はドメイン適合を実現するためのモジュールです。

CMLにおけるドメイン適合はアイテム <img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;i" title="i" /> の特徴量 <img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;x_i" title="x_i" />に対して、<img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;f(x_i;\\theta)" title="f(x_i;\theta)" /> を損失に加えることで達成します。ユーザーについても同様です。これによって埋め込みベクトルに属性情報を反映することができます。

MLPAdaptor は<img src="https://latex.codecogs.com/gif.latex?\\bg_black&space;f(x_i;\\theta)" title="f(x_i;\theta)" />に多層パーセプトロンを仮定した Adaptor クラスです。

以下のようにモデルに組み込むことができます。

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

# 開発

```bash
pip install poetry
pip install poetry-dynamic-versioning

poetry install
poetry build
# poetry lock
```

# 引用

 PyTorchCML はMIT Licenseとして使用できます。研究に用いた際は以下を引用していただけると幸いです。

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

# 参考文献

[1] Cheng-Kang Hsieh, Longqi Yang, Yin Cui, Tsung-Yi Lin, Serge Belongie, and Deborah Estrin.Collaborative metric learning. InProceedings of the 26th International Conference on World WideWeb, pp. 193–201, 2017.