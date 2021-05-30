# Unbiased Augmentation for Graph Learning

This project is a PyTorch implementation of *Unbiased Augmentation for Graph Learning* (submitted to NeurIPS 2021).

## Prerequisites

Our implementation is based on Python 3.7 and PyTorch Geometric.
Please see the full list of packages required to run our codes in `requirements.txt`.

- Python 3.7
- PyTorch 1.4.0
- PyTorch Geometric 1.6.3

PyTorch Geometric requires a separate installation process from the other packages.
We included `install.sh` to guide the installation process of PyTorch Geometric based on the OS and CUDA version.
The code includes the cases for Linux + CUDA 10.0, Linux + CUDA 10.1, and MacOS + CPU.

## Datasets

We use seven molecular datasets in our work, which are not included in this repository due to their size but can be
downloaded easily by PyTorch Geometric. 
You can run `data.py` in the `src` directory to download the datasets in the `data/graphs` directory.
Our split indices in `data/splits` are also based on these datasets.

|Name    |Graphs|  Nodes|  Edges|Features|Labels|
|:-------|-----:|------:|------:|-------:|-----:|
|DD      |  1178|  284.3| 1431.3|      89|     2|
|ENZYMES |   600|   32.6|  124.3|       3|     6|
|MUTAG   |   188|   17.9|   39.6|       7|     2|
|NCI1    |  4110|   29.9|   64.6|      37|     2|
|NCI109  |  4127|   29.7|   64.3|      38|     2|
|PROTEINS|  1113|   39.1|  145.6|       3|     2|
|PTC_MR  |   334|   14.3|   29.4|      18|     2|

## Usage

We included `demo.sh`, which reproduces the experimental results of our paper.
The code automatically downloads the datasets and trains a GIN classifier with all of our proposed approaches for graph
augmentation.
In other words, you just have to type the following command.
```
bash demo.sh
```
This demo script uses all of your GPUs by default and runs four workers for each GPU to reduce the running time.
You can change experimenal arguments such as the number of workers in `run.py` and the other hyperparameters such as
the number of epochs, batch size, or the initial learning rate in `main.py`.
Since `run.py` is a wrapper script for the parallel execution of `main.py`, all optional arguments given to `run.py` are passed also to `main.py`.
