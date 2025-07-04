import torch
import torch.nn
from data.data import get_graph_info


class GraphFeatures():
    def __init__(self,adj_path="pems04_adj.npy"):
        a = get_graph_info(adj_path)
        self.adj = a[0]
        self.in_degree = a[1]
        self.out_degree = a[2]
        self.max_in_degree = a[3]
        self.max_out_degree = a[4]