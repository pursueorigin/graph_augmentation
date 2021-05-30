"""
Our implementation of SubMix by PyTorch Geometric.
"""
import math

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import GDC
from torch_geometric.utils import subgraph, to_networkx

from augment.utils import get_sorted_neighbors, get_degrees


def to_mask(num_nodes, subset):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[subset] = 1
    return mask


def mix_labels(dst_graph, src_graph, weight, num_labels):
    out = torch.zeros(num_labels)
    out[dst_graph.y] += weight
    out[src_graph.y] += 1 - weight
    return out


def mix_graphs(dst_graph, dst_nodes, src_graph, src_nodes, num_labels, label_by='edges'):
    node_map = torch.zeros(src_graph.num_nodes, dtype=torch.long)
    node_map[src_nodes] = dst_nodes
    dst_mask = to_mask(dst_graph.num_nodes, dst_nodes)
    src_mask = to_mask(src_graph.num_nodes, src_nodes)

    edges1 = dst_graph.edge_index
    edges1 = edges1[:, ~(dst_mask[edges1[0]] & dst_mask[edges1[1]])]
    edges2 = src_graph.edge_index
    edges2 = node_map[edges2[:, src_mask[edges2[0]] & src_mask[edges2[1]]]]

    new_x = dst_graph.x.clone()
    new_x[dst_nodes] = src_graph.x[src_nodes]
    new_y = torch.zeros(num_labels)
    new_y[dst_graph.y] = 1

    new_edges = torch.cat([edges1, edges2], dim=1)
    if new_edges.size(1) == 0:
        return Data(dst_graph.x, dst_graph.edge_index, y=new_y)

    if label_by == 'nodes':
        ratio = len(dst_nodes) / dst_graph.num_nodes
    elif label_by == 'edges':
        ratio = edges1.size(1) / new_edges.size(1)
    else:
        raise ValueError(label_by)

    new_y[dst_graph.y] *= ratio
    new_y[src_graph.y] += 1 - ratio
    return Data(new_x, new_edges, y=new_y)


def gather_graphs_by_labels(graphs):
    out = []
    for i, graph in enumerate(graphs):
        y = graph.y.item()
        while y >= len(out):
            out.append([])
        out[y].append(i)
    return out


def select_graph(idx, graphs, graphs_by_labels=None):
    if graphs_by_labels is not None:
        candidates = graphs_by_labels[graphs[idx].y.item()]
        assert len(candidates) > 1
    else:
        candidates = None

    out = idx
    while out == idx:
        if candidates is not None:
            out = np.random.choice(candidates)
        else:
            out = np.random.randint(len(graphs))
    return out


def get_augment_size(dst_graph, src_graph, aug_size=0.5):
    max_size = math.ceil(min(dst_graph.num_nodes, src_graph.num_nodes) * aug_size)
    return np.random.randint(max_size)


class SubMixBase(object):
    def __init__(self, graphs, same_label=False, aug_size=0.5, label_by='edges'):
        super().__init__()
        self.graphs = graphs
        self.aug_size = aug_size
        self.num_labels = max(g.y for g in graphs) + 1
        self.graphs_by_labels = gather_graphs_by_labels(graphs) if same_label else None
        self.label_by = label_by

    def select_nodes(self, idx, size):
        return torch.randperm(self.graphs[idx].num_nodes)[:size]

    def __call__(self, idx):
        out_list = []
        for i in idx:
            j = select_graph(i, self.graphs, self.graphs_by_labels)
            graph1 = self.graphs[i]
            graph2 = self.graphs[j]

            size = get_augment_size(graph1, graph2, self.aug_size)
            nodes1 = self.select_nodes(i, size)
            nodes2 = self.select_nodes(j, size)

            out = mix_graphs(graph1, nodes1, graph2, nodes2, self.num_labels, self.label_by)
            out.y = out.y.unsqueeze(0)
            out_list.append(out)
        return out_list


class RootSelector(object):
    def __init__(self, graphs, mode='random'):
        assert mode in ['random', 'positional', 'important']
        self.graphs = graphs
        self.mode = mode

        if mode == 'positional':
            self.degrees = []
            for graph in graphs:
                self.degrees.append(get_degrees(graph))
        elif mode == 'important':
            self.degree_dists = []
            for graph in graphs:
                d = get_degrees(graph).numpy()
                self.degree_dists.append(d / d.sum())

    def get_positional_target(self, idx, position):
        num_nodes = self.graphs[idx].num_nodes
        order = (self.degrees[idx] + torch.rand(num_nodes)).argsort()
        return order[int(position * num_nodes)]

    def __call__(self, idx1, idx2):
        if self.mode == 'random':
            target1 = torch.randint(self.graphs[idx1].num_nodes, size=(1,))
            target2 = torch.randint(self.graphs[idx2].num_nodes, size=(1,))
        elif self.mode == 'positional':
            position = torch.rand(size=(1,))
            target1 = self.get_positional_target(idx1, position)
            target2 = self.get_positional_target(idx2, position)
        elif self.mode == 'important':
            target1 = torch.randint(self.graphs[idx1].num_nodes, size=(1,))
            target2 = np.random.choice(self.graphs[idx2].num_nodes, p=self.degree_dists[idx2])
        else:
            raise ValueError(self.mode)
        return target1, target2


class SubMix(object):
    def __init__(self, graphs, aug_size=0.5, root='random', norm='sym', same_label=False,
                 shuffle=False, label_by='edges'):
        super().__init__()
        assert norm in ['sym', 'col']

        self.graphs = graphs
        self.num_labels = max(g.y for g in graphs) + 1
        self.graphs_by_labels = gather_graphs_by_labels(graphs) if same_label else None
        self.root_selector = RootSelector(graphs, root)
        self.shuffle = shuffle
        self.aug_size = aug_size
        self.label_by = label_by

        model = GDC(normalization_in=norm, normalization_out=None,
                    sparsification_kwargs=dict(method='threshold', eps=0))
        neighbors = []
        for graph in graphs:
            new_graph = Data(graph.x, graph.edge_index, y=graph.y)
            model(new_graph)
            neighbors.append(get_sorted_neighbors(new_graph))
        self.neighbors = neighbors

    def select_nodes(self, idx, root, size):
        return self.neighbors[idx][root][:size]

    def __call__(self, indices):
        out_list = []
        for i in indices:
            j = select_graph(i, self.graphs, self.graphs_by_labels)
            graph1 = self.graphs[i]
            graph2 = self.graphs[j]

            size = get_augment_size(graph1, graph2, self.aug_size)
            root1, root2 = self.root_selector(i, j)
            nodes1 = self.select_nodes(i, root1, size)
            nodes2 = self.select_nodes(j, root2, size)
            if self.shuffle:
                nodes2 = nodes2[torch.randperm(len(nodes2))]
            out = mix_graphs(graph1, nodes1, graph2, nodes2, self.num_labels, self.label_by)
            out.y = out.y.unsqueeze(0)
            out_list.append(out)
        return out_list

    def describe(self, index, check_connectivity=False):
        j = select_graph(index, self.graphs, self.graphs_by_labels)
        graph1 = self.graphs[index]
        graph2 = self.graphs[j]

        size = get_augment_size(graph1, graph2, self.aug_size)
        root1, root2 = self.root_selector(index, j)
        nodes1 = self.select_nodes(index, root1, size)
        nodes2 = self.select_nodes(j, root2, size)
        if self.shuffle:
            nodes2 = nodes2[torch.randperm(len(nodes2))]
        graph3 = mix_graphs(graph1, nodes1, graph2, nodes2, self.num_labels, self.label_by)
        graph3.y = graph3.y.unsqueeze(0)

        if check_connectivity:
            import networkx as nx

            graph2_nx = to_networkx(graph2, to_undirected=True)
            edges4, _ = subgraph(nodes2, graph2.edge_index, relabel_nodes=True, num_nodes=graph2.num_nodes)
            graph4 = Data(edge_index=edges4, num_nodes=len(nodes2))
            if graph4.num_edges > 0:
                graph4_nx = to_networkx(graph4, to_undirected=True)
                if not nx.is_connected(graph4_nx):
                    if len(nodes2) <= len(nx.node_connected_component(graph2_nx, root2.item())):
                        raise ValueError('Not connected')

        return graph1, graph2, root1, root2, nodes1, nodes2, graph3
