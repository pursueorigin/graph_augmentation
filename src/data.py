"""
Helper functions to download and process our graph datasets.
"""
import json
import os

from sklearn.model_selection import StratifiedKFold
from torch_geometric.datasets import TUDataset
import numpy as np
from torch_geometric.utils import contains_self_loops, contains_isolated_nodes, is_undirected

ROOT = '../data'
DATASETS = ['DD', 'ENZYMES', 'MUTAG', 'NCI1', 'NCI109', 'PROTEINS', 'PTC_MR']


def load_data(dataset):
    assert dataset in DATASETS
    data = TUDataset(root=os.path.join(ROOT, 'graphs'), name=dataset, use_node_attr=False)
    data.data.edge_attr = None
    return data


def load_data_fold(dataset, fold, num_folds=10, seed=0):
    assert 0 <= fold < 10

    data = load_data(dataset)
    path = os.path.join(ROOT, 'splits', dataset, f'{fold}.json')
    if not os.path.exists(path):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        trn_idx, test_idx = list(skf.split(np.zeros(data.len()), data.data.y))[fold]
        trn_idx = [int(e) for e in trn_idx]
        test_idx = [int(e) for e in test_idx]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(training=trn_idx, test=test_idx), f, indent=4)

    with open(path) as f:
        indices = json.load(f)
    trn_graphs = [data[i] for i in indices['training']]
    test_graphs = [data[i] for i in indices['test']]
    return trn_graphs, test_graphs


def print_stats():
    for data in DATASETS:
        out = load_data(data)
        num_graphs = len(out)
        avg_nodes = out.data.x.size(0) / num_graphs
        avg_edges = out.data.edge_index.size(1) / num_graphs
        num_features = out.num_features
        num_classes = out.num_classes
        print(f'{data}\t{num_graphs}\t{avg_nodes}\t{avg_edges}\t{num_features}\t{num_classes}',
              end='\t')

        undirected, self_loops, isolated_nodes, onehot = True, False, False, True
        for graph in out:
            if not is_undirected(graph.edge_index, num_nodes=graph.num_nodes):
                undirected = False
            if contains_self_loops(graph.edge_index):
                self_loops = True
            if contains_isolated_nodes(graph.edge_index, num_nodes=graph.num_nodes):
                isolated_nodes = True
            if ((graph.x > 0).sum(dim=1) != 1).sum() > 0:
                onehot = False
        print(f'{undirected}\t{self_loops}\t{isolated_nodes}\t{onehot}')


def download():
    for data in DATASETS:
        load_data(data)


if __name__ == '__main__':
    download()
