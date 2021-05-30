"""
A wrapper class to augment a set of graphs by a specific algorithm.
"""
import inspect
from torch_geometric.data import Batch

from augment.merge import MergeEdge
from augment.split import SplitNode
from augment.submix import SubMixBase, SubMix
from augment.nodesam import NodeSamBase, NodeSam


class Augment(object):
    def __init__(self, graphs, method=None, **kwargs):
        super().__init__()
        self.graphs = graphs
        if method is None:
            self.model = None
        else:
            model_class = eval(method)
            parameters = inspect.signature(model_class).parameters
            args = {k: v for k, v in kwargs.items() if k in parameters}
            self.model = model_class(graphs, **args)

    def augment(self, idx):
        if self.model is None:
            return [self.graphs[i] for i in idx]
        else:
            return self.model(idx)

    def __call__(self, idx, as_batch=True):
        data = self.augment(idx)
        if as_batch:
            data = Batch().from_data_list(data)
        return data
