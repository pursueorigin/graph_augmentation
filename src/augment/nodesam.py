"""
Our implementation of NodeSam, which combines the merge and split operations.
"""
from augment.merge import MergeEdge
from augment.split import SplitNode


class NodeSam(object):
    def __init__(self, graphs, split_weighted='none', merge_weighted='none', adjustment=True):
        self.split = SplitNode(graphs, weighted=split_weighted, adjustment=adjustment)
        self.merge = MergeEdge(graphs, weighted=merge_weighted)

    def __call__(self, idx):
        return self.merge(graphs=self.split(idx))

    def describe(self, index):
        graph1 = self.split([index])[0]
        node1 = self.split.selected_node
        node2 = self.split.generated_node
        graph2 = self.merge(graphs=[graph1])[0]
        nodes = self.merge.selected_nodes
        return graph1, graph2, node1, node2, nodes


class NodeSamBase(NodeSam):
    def __init__(self, graphs, split_weighted='none', merge_weighted='none'):
        super().__init__(graphs, split_weighted, merge_weighted, adjustment=False)
