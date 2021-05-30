"""
Our implementation of the merge operation of NodeSam by PyTorch Geometric.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, remove_self_loops

from augment.utils import to_directed


def merge_nodes(graph, nodes, onehot=False):
    assert len(nodes) == 2

    winner = nodes[0]
    losers = nodes[1:]
    old_nodes = graph.num_nodes
    new_nodes = graph.num_nodes - len(nodes) + 1
    edge_index = graph.edge_index

    # Remove duplicate edges when the selected nodes make triangles.
    mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
    mask[edge_index[1, edge_index[0] == nodes[0]]] = True
    duplicates1 = (edge_index[0] == nodes[1]) & mask[edge_index[1]]
    duplicates2 = (edge_index[1] == nodes[1]) & mask[edge_index[0]]
    edge_index = edge_index[:, ~(duplicates1 | duplicates2)]

    node_mask = torch.ones(old_nodes, dtype=torch.bool)
    node_mask[losers] = 0
    node_map = torch.zeros(old_nodes, dtype=torch.int64)
    node_map[node_mask] = torch.arange(new_nodes)
    winner = node_map[winner]  # in case winner is remapped.
    node_map[losers] = winner

    new_x = torch.zeros(new_nodes, graph.num_features)
    new_x.index_add_(0, node_map, graph.x)
    new_x[winner] /= new_x[winner].sum()

    if onehot and (new_x[winner] > 0).sum() > 1:
        selected = np.random.choice(graph.num_features, p=new_x[winner])
        new_x[winner] = 0
        new_x[winner, selected] = 1

    edge_index = node_map[edge_index]
    edge_index, _ = remove_self_loops(edge_index)
    return Data(x=new_x, edge_index=edge_index, y=graph.y)


def get_scores(graph, edges=None):
    if edges is None:
        edges = graph.edge_index
    degrees = degree(graph.edge_index[0], graph.num_nodes)
    scores = []
    for src, dst in edges.t().numpy():
        scores.append(1 / (degrees[src] * degrees[dst]))
    scores = np.array(scores)
    return scores / scores.sum()


class MergeEdge(object):
    def __init__(self, graphs, same_attr=False, onehot=False, weighted='none'):
        super().__init__()
        self.graphs = graphs
        self.onehot = onehot
        self.same_attr = same_attr
        self.weighted = weighted

        self.candidates = [self.get_candidates(g) for g in graphs]
        self.scores = []
        for graph, candi in zip(graphs, self.candidates):
            self.scores.append(self.get_scores(graph, candi))

        self.selected_nodes = None

    def get_candidates(self, graph):
        edges = to_directed(graph.edge_index)
        if self.same_attr:
            attributes = graph.x.argmax(dim=1)[edges]
            edges = edges[:, attributes[0] == attributes[1]]
        return edges

    def get_scores(self, graph, candidates):
        if self.weighted == 'degree':
            return get_scores(graph, candidates)
        else:
            return None

    def run(self, graph, candidates, scores):
        num_candidates = candidates.size(1)
        if num_candidates == 0:
            return graph
        else:
            target = np.random.choice(num_candidates, p=scores)
            nodes = candidates[:, target]
            self.selected_nodes = nodes
            return merge_nodes(graph, nodes, onehot=self.onehot)

    def __call__(self, indices=None, graphs=None):
        data = []
        if indices is not None:
            for i in indices:
                data.append((self.graphs[i], self.candidates[i], self.scores[i]))
        elif graphs is not None:
            for graph in graphs:
                candi = self.get_candidates(graph)
                score = self.get_scores(graph, candi)
                data.append((graph, candi, score))

        out_list = []
        for graph, candi, score in data:
            out_list.append(self.run(graph, candi, score))
        return out_list
