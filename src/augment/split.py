"""
Our implementation of the split operation of NodeSam by PyTorch Geometric.
"""
import math

import torch
import numpy as np
from torch_geometric.data import Data

from augment.utils import to_directed, to_undirected, get_degrees, get_neighbors


def split(array, p=0.5):
    array = array.copy()
    np.random.shuffle(array)
    n = np.random.binomial(len(array), p)
    return array[:n], array[n:]


class SplitNode(object):
    def __init__(self, graphs, weighted='none', adjustment=False):
        super().__init__()
        self.weighted = weighted
        self.adjustment = adjustment
        self.graphs = graphs
        self.directed_edges = [to_directed(graph.edge_index).t().numpy() for graph in graphs]
        self.neighbors = [get_neighbors(graph) for graph in graphs]
        self.degree_dist = [None for _ in range(len(graphs))]

        self.triangles, self.tri_nodes, self.degrees = [], [], []
        if adjustment:
            for i in range(len(graphs)):
                triangles, tri_nodes = self.find_triangles(i)
                self.degrees.append(get_degrees(graphs[i]))
                self.triangles.append(triangles)
                self.tri_nodes.append(tri_nodes)

        self.selected_node = None
        self.generated_node = None

    def find_triangles(self, idx):
        triangles = [0 for _ in range(self.graphs[idx].num_nodes)]
        tri_nodes = [[] for _ in range(self.graphs[idx].num_nodes)]
        neighbor_sets = [set(nn) for nn in self.neighbors[idx]]
        for n1, n2 in self.directed_edges[idx]:
            for n3 in neighbor_sets[n1].intersection(neighbor_sets[n2]):
                triangles[n3] += 1
                tri_nodes[n3].append(n1)
                tri_nodes[n3].append(n2)
        tri_nodes = [np.unique(np.array(nn, dtype=np.int64)) for nn in tri_nodes]
        return triangles, tri_nodes

    def select_node(self, idx):
        num_nodes = self.graphs[idx].num_nodes
        if self.weighted == 'degree':
            if self.degree_dist[idx] is None:
                degree_dist = get_degrees(self.graphs[idx]).numpy()
                self.degree_dist[idx] = degree_dist / degree_dist.sum()
            return np.random.choice(num_nodes, p=self.degree_dist[idx])
        elif self.weighted == 'none':
            return np.random.randint(num_nodes)
        else:
            raise ValueError()

    def split_with_adjustment(self, idx, node):
        v = self.graphs[idx].num_nodes
        e = self.graphs[idx].num_edges // 2
        t = self.triangles[idx][node]
        d = self.degrees[idx][node]
        c = e - 2 - 3 * t / d
        n = (math.sqrt(c ** 2 + 4 * t * v - 6 * t) - c) / 2

        tri_nodes = self.tri_nodes[idx][node]
        tri_nodes, _ = split(tri_nodes, p=min(n / len(tri_nodes), 1))
        non_tri_nodes = np.setdiff1d(self.neighbors[idx][node], tri_nodes, assume_unique=True)
        nset1, nset2 = split(non_tri_nodes)
        nset1 = np.concatenate([nset1, tri_nodes])
        nset2 = np.concatenate([nset2, tri_nodes])
        return nset1, nset2

    def __call__(self, idx):
        out_list = []
        for i in idx:
            graph = self.graphs[i]
            old_node = self.select_node(i)
            new_node = graph.num_nodes
            self.selected_node = old_node
            self.generated_node = new_node

            if self.adjustment and self.triangles[i][old_node] > 0:
                nset1, nset2 = self.split_with_adjustment(i, old_node)
            else:
                nset1, nset2 = split(self.neighbors[i][old_node])

            edges = self.directed_edges[i]
            edge_index = [[n1, n2] for n1, n2 in edges if n1 != old_node and n2 != old_node]
            for n in nset1:
                edge_index.append([n, old_node])
            for n in nset2:
                edge_index.append([n, new_node])
            edge_index.append([old_node, new_node])
            edge_index = to_undirected(torch.tensor(edge_index).t())

            new_x = torch.cat([graph.x, graph.x[old_node].view(1, -1)])
            out_list.append(Data(x=new_x, edge_index=edge_index, y=graph.y))
        return out_list
