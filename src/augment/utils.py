"""
Helper functions for our graph augmentation approaches.
"""
import torch
import numpy as np
from torch_geometric.utils import degree


def get_degrees(graph):
    return degree(graph.edge_index[0], num_nodes=graph.num_nodes)


def to_directed(edge_index):
    return edge_index[:, edge_index[0] < edge_index[1]]


def to_undirected(edge_index):
    assert (edge_index[0] == edge_index[1]).sum() == 0  # no self-loops
    return torch.cat([edge_index, edge_index.flip([0])], dim=1)


def get_neighbors(graph, dim=0, as_tensor=False, self_loops=False):
    neighbors = [[] for _ in range(graph.num_nodes)]

    if self_loops:
        for n in range(graph.num_nodes):
            neighbors[n].append(n)

    for row, col in graph.edge_index.t().numpy():
        if dim == 0:
            neighbors[col].append(row)
        elif dim == 1:
            neighbors[row].append(col)
        else:
            raise ValueError()

    if as_tensor:
        return [torch.tensor(s) for s in neighbors]
    else:
        return [np.array(s) for s in neighbors]


def get_sorted_neighbors(graph, dim=0):
    neighbors = [[] for _ in range(graph.num_nodes)]
    for (row, col), score in zip(graph.edge_index.t().numpy(), graph.edge_attr):
        if dim == 0:
            neighbors[col].append((score.item(), row))
        elif dim == 1:
            neighbors[row].append((score.item(), col))
        else:
            raise ValueError()
    for i in range(len(neighbors)):
        neighbors[i] = [v for (u, v) in sorted(neighbors[i], reverse=True)]
    return [torch.tensor(s) for s in neighbors]
