from torch_geometric.utils import from_networkx
import networkx as nx

def dgl_to_pyg_data(dgl_graph):
    nx_graph = dgl_graph.to_networkx(node_attrs=['feat', 'label', 'train_mask', 'val_mask', 'test_mask'])
    pyg_data = from_networkx(nx_graph)
    pyg_data.x = pyg_data.feat
    pyg_data.y = pyg_data.label
    pyg_data.train_mask = pyg_data.train_mask
    pyg_data.val_mask = pyg_data.val_mask
    pyg_data.test_mask = pyg_data.test_mask
    return pyg_data

