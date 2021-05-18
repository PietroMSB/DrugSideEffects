from spektral.data.dataset import Dataset
from spektral.data.graph import Graph
import numpy as np
from GNNv3.GNN.graph_class import GraphObject


class CustomDataset(Dataset):
    """
    A custom dataset from list of GraphObjects.
    """

    def __init__(self, g, **kwargs):
        self.graphs = g
        super().__init__(**kwargs)

    def read(self):
        return [Graph(a=g.Adjacency, e=g.arcs[:,2:], x=g.nodes, y=g.targets) for g in self.graphs]

'''for g in out:
    e = np.zeros((g.n_nodes, g.n_nodes, g.n_edge_features), dtype=np.float32)
    e[g.a.row, g.a.col, :] = g.e
    g.e = e'''
