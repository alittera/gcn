# Simplified Graph Convolutional Networks

This is a TensorFlow implementation of Simplified Graph Convolutional Networks for the task of (semi-supervised) classification of nodes in a graph carried out as a project for the examination of Neural Networks, at Sapienza university of Rome.
 
The project is based on:

Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907)

Wu , Zhang ,de Souza Jrm, [Simplifying Graph Convolutional Networks](https://arxiv.org/pdf/1902.07153)


## Installation

```bash
python setup.py install
```

## Requirements
* tensorflow (>0.12)
* networkx

## Run the demo

```bash
cd gcn
python train.py -model sgcn -dataset pubmed
```

## Data

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://www.cs.umd.edu/~sen/lbc-proj/LBC.html. In our version (see `data` folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861), ICML 2016). 

You can specify a dataset as follows:

```bash
python train.py --dataset citeseer
python train.py --dataset cora
python train.py --dataset pubmed
```

(or by editing `train.py`)

## Models

You can choose between the following models: 
* `gcn`: Graph convolutional network (Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907), 2016)
* `gcn_cheby`: Chebyshev polynomial version of graph convolutional network as described in (Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NIPS 2016)
* `sgcn`: [Simplifying Graph Convolutional Networks](https://arxiv.org/pdf/1902.07153)
* `dense`: Basic multi-layer perceptron that supports sparse inputs
