import copy
import json
import math
import os
import pickle as pkl
import random
import sys
import time

import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy.sparse as sp
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset, CoauthorCSDataset, CoauthorPhysicsDataset, \
    RedditDataset, WikiCSDataset, AmazonRatingsDataset, QuestionsDataset, RomanEmpireDataset, FlickrDataset, \
    CoraFullDataset
from dgl.data import citation_graph as citegrh
from dgl.nn.pytorch import GraphConv
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch_geometric.datasets import CitationFull
from tqdm import tqdm
from pygip.models.attack.base import BaseAttack
from pygip.models.nn import GCN, GraphSAGE
from pygip.utils.metrics import AttackMetric, AttackCompMetric
import pygip.utils.metrics as metrics

time_limit = 300


def get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score, adj_matrix2):
    receptive_vector = ((cur_neighbors + adj_matrix2[selected_node]) != 0) + 0
    count = weighted_score.dot(receptive_vector)
    return count


def get_current_neighbors_dense(cur_nodes, adj_matrix2):
    if np.array(cur_nodes).shape[0] == 0:
        return 0
    neighbors = (adj_matrix2[list(cur_nodes)].sum(axis=0) != 0) + 0
    return neighbors


def get_current_neighbors_1(cur_nodes, adj_matrix):
    if np.array(cur_nodes).shape[0] == 0:
        return 0
    # mirror behavior of get_current_neighbors_dense for non-dense adjacency
    neighbors = (adj_matrix[list(cur_nodes)].sum(axis=0) != 0) + 0
    return neighbors


def get_entropy_contribute(npy_m1, npy_m2):
    entro1 = 0
    entro2 = 0
    for i in range(npy_m1.shape[0]):
        entro1 -= np.sum(npy_m1[i] * np.log2(npy_m1[i]))
        entro2 -= np.sum(npy_m2[i] * np.log2(npy_m2[i]))
    return entro1 - entro2


def get_max_info_entropy_node_set(idx_used,
                                  high_score_nodes,
                                  labels,
                                  batch_size,
                                  adj_matrix2,
                                  num_class,
                                  model_prediction):
    max_info_node_set = []
    high_score_nodes_ = copy.deepcopy(high_score_nodes)
    labels_ = copy.deepcopy(labels)
    for k in range(batch_size):
        score_list = np.zeros(len(high_score_nodes_))
        for i in range(len(high_score_nodes_)):
            labels_tmp = copy.deepcopy(labels_)
            node = high_score_nodes_[i]
            node_neighbors = get_current_neighbors_dense([node], adj_matrix2)
            adj_neigh = adj_matrix2[list(node_neighbors)]
            aay = np.matmul(adj_neigh, labels_)
            total_score = 0
            for j in range(num_class):
                if model_prediction[node][j] != 0:
                    labels_tmp[node] = 0
                    labels_tmp[node][j] = 1
                    aay_ = np.matmul(adj_neigh, labels_tmp)
                    total_score += model_prediction[node][j] * get_entropy_contribute(aay, aay_)
            score_list[i] = total_score
        idx = np.argmax(score_list)
        max_node = high_score_nodes_[idx]
        max_info_node_set.append(max_node)
        labels_[max_node] = model_prediction[max_node]
        high_score_nodes_.remove(max_node)
    return max_info_node_set


def get_max_nnd_node_dense(idx_used,
                           high_score_nodes,
                           min_distance,
                           distance_aax,
                           num_ones,
                           num_node,
                           adj_matrix2,
                           gamma=1):
    dmax = np.ones(num_node)

    max_receptive_node = 0
    max_total_score = 0
    cur_neighbors = get_current_neighbors_dense(idx_used, adj_matrix2)
    for node in high_score_nodes:
        receptive_field = get_receptive_fields_dense(cur_neighbors, node, num_ones, adj_matrix2)
        node_distance = distance_aax[node, :]
        node_distance = np.where(node_distance < min_distance, node_distance, min_distance)
        node_distance = dmax - node_distance
        distance_score = node_distance.dot(num_ones)
        total_score = receptive_field / num_node + gamma * distance_score / num_node
        if total_score > max_total_score:
            max_total_score = total_score
            max_receptive_node = node
    return max_receptive_node


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def compute_distance(_i, _j, features_aax):
    return la.norm(features_aax[_i, :] - features_aax[_j, :])


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data_from_grain(path="./data", dataset="cora"):
    """
    ind.[:dataset].x     => the feature vectors of the training instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].y     => the one-hot labels of the labeled training instances (numpy.ndarray)
    ind.[:dataset].allx  => the feature vectors of both labeled and unlabeled training instances (csr_matrix)
    ind.[:dataset].ally  => the labels for instances in ind.dataset_str.allx (numpy.ndarray)
    ind.[:dataset].graph => the dict in the format {index: [index of neighbor nodes]} (collections.defaultdict)
    ind.[:dataset].tx => the feature vectors of the test instances (scipy.sparse.csr.csr_matrix)
    ind.[:dataset].ty => the one-hot labels of the test instances (numpy.ndarray)
    ind.[:dataset].test.index => indices of test instances in graph, for the inductive setting
    """
    print("\n[STEP 1]: Upload {} dataset.".format(dataset))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(path, dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(path, dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Citeseer dataset contains some isolated nodes in the graph
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    print("| # of nodes : {}".format(adj.shape[0]))
    print("| # of edges : {}".format(adj.sum().sum() / 2))

    features = normalize(features)
    print("| # of features : {}".format(features.shape[1]))
    print("| # of clases   : {}".format(ally.shape[1]))

    features = torch.FloatTensor(np.array(features.todense()))
    sparse_mx = adj.tocoo().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    if dataset == 'citeseer':
        save_label = np.where(labels)[1]
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)
    idx_test = test_idx_range.tolist()

    print("| # of train set : {}".format(len(idx_train)))
    print("| # of val set   : {}".format(len(idx_val)))
    print("| # of test set  : {}".format(len(idx_test)))

    idx_train, idx_val, idx_test = list(map(lambda x: torch.LongTensor(x), [idx_train, idx_val, idx_test]))

    def missing_elements(L):
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    if dataset == 'citeseer':
        L = np.sort(idx_test)
        missing = missing_elements(L)

        for element in missing:
            save_label = np.insert(save_label, element, 0)

        labels = torch.LongTensor(save_label)

    return adj, features, labels, idx_train, idx_val, idx_test


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_normalized_adjacency(adj):
    adj = adj  # + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


class GCN_drop(nn.Module):
    def __init__(self, feature_number, label_number, dropout=0.85, nhid=128):
        super(GCN_drop, self).__init__()

        self.gc1 = GraphConv(feature_number, nhid, bias=True)
        self.gc2 = GraphConv(nhid, label_number, bias=True)
        self.dropout = dropout

    def forward(self, g, features):
        x = F.dropout(features, self.dropout, training=self.training)
        x = F.relu(self.gc1(g, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(g, x)
        return x


def convert_pyg_to_dgl(pyg_data):
    """
    Converts a PyTorch Geometric Data object into a DGLGraph.

    Args:
        pyg_data (torch_geometric.data.Data): PyTorch Geometric Data object.

    Returns:
        dgl.DGLGraph: The converted DGL graph.
    """
    edge_index = pyg_data.edge_index
    num_nodes = pyg_data.num_nodes

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

    if hasattr(pyg_data, 'x') and pyg_data.x is not None:
        g.ndata['feat'] = pyg_data.x

    if hasattr(pyg_data, 'y') and pyg_data.y is not None:
        g.ndata['label'] = pyg_data.y

    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(pyg_data, mask_name) and getattr(pyg_data, mask_name) is not None:
            g.ndata[mask_name] = getattr(pyg_data, mask_name)

    return g


def load_data(dataset_name):
    """
    Load a dataset by name and return a DGL graph and its associated tensors.
    This function normalizes the input name and handles the special case for
    PyG datasets (e.g. dblp) which need conversion to DGL.
    Returns: (g, features, labels, num_nodes, train_mask, test_mask)
    """
    name = (dataset_name or '').lower()
    data = None

    if name == 'cora':
        data = citegrh.load_cora()
    elif name == 'citeseer':
        data = citegrh.load_citeseer()
    elif name == 'pubmed':
        data = citegrh.load_pubmed()
    elif name == 'amazoncomputer':
        data = AmazonCoBuyComputerDataset()
    elif name == 'amazonphoto':
        data = AmazonCoBuyPhotoDataset()
    elif name in ('coauthorcs', 'coauthor_cs'):
        data = CoauthorCSDataset()
    elif name in ('coauthorphysics', 'coauthor_physics'):
        data = CoauthorPhysicsDataset()
    elif name == 'reddit':
        data = RedditDataset()
    elif name in ('wiki', 'wikics'):
        data = WikiCSDataset()
    elif name == 'amazonrating':
        data = AmazonRatingsDataset()
    elif name == 'question':
        data = QuestionsDataset()
    elif name == 'roman':
        data = RomanEmpireDataset()
    elif name == 'flickr':
        data = FlickrDataset()
    elif name in ('cora_full', 'corafull'):
        data = CoraFullDataset()
    elif name == 'dblp':
        data = CitationFull(root='./data/', name='DBLP')
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}' in CEGA.load_data")

    # Convert PyG dataset to DGL when necessary
    if name == 'dblp':
        pyg_data = data[0]
        g = convert_pyg_to_dgl(pyg_data)
    else:
        g = data[0]

    # remove isolated nodes if any
    isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
    if isolated_nodes.numel() > 0:
        g.remove_nodes(isolated_nodes)

    # Extract features/labels/masks in a consistent manner
    if name in ['cora', 'citeseer', 'pubmed', 'reddit', 'flickr']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        test_mask = g.ndata['test_mask']
        num_nodes = g.num_nodes()
    elif name == 'wiki':
        features = g.ndata['feat']
        labels = g.ndata['label']
        test_mask = g.ndata['test_mask'].bool()
        train_mask = (~g.ndata['test_mask']).bool()
        num_nodes = g.num_nodes()
    elif name in ['amazoncomputer', 'amazonphoto', 'coauthorcs', 'coauthorphysics', 'cora_full', 'dblp']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        num_nodes = g.num_nodes()
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        torch.manual_seed(42)
        indices = torch.randperm(num_nodes)
        num_train = int(num_nodes * 0.6)
        train_mask[indices[:num_train]] = True
        test_mask[indices[num_train:]] = True
    elif name in ['amazonrating', 'question', 'roman']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        num_nodes = g.num_nodes()
        train_mask = g.ndata['train_mask'][:, 0]
        test_mask = g.ndata['test_mask'][:, 0]
    else:
        # best-effort fallback
        features = g.ndata.get('feat', None)
        labels = g.ndata.get('label', None)
        num_nodes = g.num_nodes()
        train_mask = g.ndata.get('train_mask', torch.zeros(num_nodes, dtype=torch.bool))
        test_mask = g.ndata.get('test_mask', torch.ones(num_nodes, dtype=torch.bool))

    return g, features, labels, num_nodes, train_mask, test_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def aug_normalized_adjacency(adj):
    adj = adj  # + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()


class CEGA(BaseAttack):
    """Lightweight CEGA wrapper implementing the expected BaseAttack API.

    This implementation provides a minimal, well-behaved CEGA class so that
    the attack registry can import and instantiate it. It trains or loads a
    target GCN and performs a simple model-extraction surrogate training using
    the target's predictions as pseudolabels. The goal is correctness and
    interoperability rather than reproducing the full original CEGA algorithm.
    """
    supported_api_types = {"dgl"}

    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction, model_path)
        # graph tensors on device
        self.graph = dataset.graph_data.to(self.device)
        self.features = self.graph.ndata['feat']
        self.labels = self.graph.ndata['label']
        self.train_mask = self.graph.ndata.get('train_mask')
        self.test_mask = self.graph.ndata.get('test_mask')

        # metadata
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # Ensure we always have boolean train/test masks on the correct device.
        # Some datasets in DGL expose masks as index tensors or missing masks;
        # create a sensible fallback split if needed.
        try:
            # if masks are missing, synthesize a 60/40 train/test split
            if (self.train_mask is None) or (self.test_mask is None):
                torch.manual_seed(42)
                num_nodes = int(self.num_nodes)
                indices = torch.randperm(num_nodes)
                num_train = int(num_nodes * 0.6)
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                train_mask[indices[:num_train]] = True
                test_mask = ~train_mask
                self.train_mask = train_mask.to(self.device)
                self.test_mask = test_mask.to(self.device)
            else:
                # convert index-style masks to boolean and move to device
                if self.train_mask.dtype != torch.bool:
                    # handle index or other mask formats
                    try:
                        # if mask is a list/1D indices tensor
                        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device).index_fill_(0, self.train_mask.to(self.device), True)
                    except Exception:
                        self.train_mask = self.train_mask.to(self.device).bool()
                else:
                    self.train_mask = self.train_mask.to(self.device)

                if self.test_mask.dtype != torch.bool:
                    try:
                        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device).index_fill_(0, self.test_mask.to(self.device), True)
                    except Exception:
                        self.test_mask = self.test_mask.to(self.device).bool()
                else:
                    self.test_mask = self.test_mask.to(self.device)
        except Exception:
            # fallback: create simple split on device
            torch.manual_seed(42)
            num_nodes = int(self.num_nodes)
            indices = torch.randperm(num_nodes)
            num_train = int(num_nodes * 0.6)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[indices[:num_train]] = True
            test_mask = ~train_mask
            self.train_mask = train_mask.to(self.device)
            self.test_mask = test_mask.to(self.device)

        # prepare target model
        if self.model_path is None:
            self._train_target_model()
        else:
            self._load_model(self.model_path)

    def _load_model(self, model_path):
        """Load a pre-trained target model state dict."""
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        self.net1.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net1.eval()

    def _forward(self, model, graph, features):
        """Lightweight forward wrapper that supports GCN and GraphSAGE-style call signatures.

        GraphSAGE implementations sometimes expect a list of blocks ([g, g]) as the first
        argument; this helper tries the normal call and falls back to the two-block format
        if a TypeError arises. This mirrors the wrapper used in DataFreeMEA.
        """
        # prefer explicit type-check if GraphSAGE is available
        try:
            if isinstance(model, GraphSAGE):
                return model([graph, graph], features)
        except Exception:
            # isinstance checks may fail if GraphSAGE isn't available in the runtime
            pass

        # Generic attempt: try normal call and fall back on the two-block input
        try:
            return model(graph, features)
        except TypeError:
            return model([graph, graph], features)

    def _train_target_model(self, epochs: int = 200, lr: float = 0.01):
        """Train a simple GCN target model on the provided dataset.

        Parameters are configurable so callers (and the benchmarking grid) can
        adjust training duration and learning rate.
        """
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=lr, weight_decay=5e-4)
        start_time = time.time()
        for epoch in range(int(epochs)):
            self.net1.train()
            logits = self.net1(self.graph, self.features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.net1.eval()
        # publish actual elapsed training time so AttackCompMetric created later can capture it
        try:
            elapsed = time.time() - start_time
            metrics.LAST_TARGET_TRAIN_TIME = float(elapsed)
        except Exception:
            # fall back to zero when metric module is unavailable
            try:
                metrics.LAST_TARGET_TRAIN_TIME = 0.0
            except Exception:
                pass
        return self.net1

    def attack(self, **kwargs):
        """Perform a lightweight extraction: query the target and train a surrogate.

        Accepts runtime parameters via kwargs. Supported keys:
        - epochs_per_cycle: fallback epoch count for both target & surrogate
        - target_epochs / surrogate_epochs: override respective training lengths
        - LR_CEGA or lr: learning rate for target/surrogate (surrogate_lr can override)
        - surrogate_lr: learning rate specifically for surrogate
        - num_query / budget_nodes: number of nodes to query from victim (optional)
        """
        try:
            # Read configuration from kwargs with sensible fallbacks
            epochs_per_cycle = int(kwargs.get('epochs_per_cycle', 200))
            epochs_target = int(kwargs.get('target_epochs', epochs_per_cycle))
            epochs_surrogate = int(kwargs.get('surrogate_epochs', epochs_per_cycle))
            lr_c = float(kwargs.get('LR_CEGA', kwargs.get('lr', 0.01)))
            surrogate_lr = float(kwargs.get('surrogate_lr', lr_c))

            # If user explicitly requests retraining target with provided epochs/lr,
            # do it; otherwise the model trained during __init__ remains.
            if kwargs.get('retrain_target', False):
                self._train_target_model(epochs=epochs_target, lr=lr_c)

            # Prepare metric containers and timers
            metric = AttackMetric()
            metric_comp = AttackCompMetric()
            # ensure timer starts immediately so total_time/gpu_hours are recorded
            metric_comp.start()
            attack_start = time.time()
            total_query_time = 0.0

            # Ensure we have a consistent victim model reference
            self.model = getattr(self, 'net1', getattr(self, 'model', None))
            if self.model is None:
                self._train_target_model(epochs=epochs_target, lr=lr_c)
                self.model = self.net1

            # Choose a limited set of query/budget nodes. Priority: explicit kwargs -> attack_node_fraction -> default 20*C
            num_query = None
            if 'num_query' in kwargs:
                num_query = int(kwargs.get('num_query'))
            elif 'budget_nodes' in kwargs:
                num_query = int(kwargs.get('budget_nodes'))
            elif self.attack_node_fraction is not None:
                num_query = max(1, int(self.num_nodes * float(self.attack_node_fraction)))
            else:
                num_query = 20 * int(self.num_classes)

            # Build candidate pool: prefer unlabeled/non-train nodes; fallback to all nodes
            try:
                candidate_mask = (~self.train_mask).to(torch.bool) if hasattr(self, 'train_mask') and self.train_mask is not None else None
                if candidate_mask is not None and candidate_mask.any():
                    candidates = candidate_mask.nonzero().view(-1)
                else:
                    candidates = torch.arange(self.num_nodes, device=self.device)
            except Exception:
                candidates = torch.arange(self.num_nodes, device=self.device)

            # Class-balanced sampling when possible
            query_indices = []
            per_class = max(1, num_query // max(1, int(self.num_classes)))
            labels = self.labels
            rng = random.Random(kwargs.get('seed', None))

            for c in range(int(self.num_classes)):
                # indices in candidates with class c
                cand_c = candidates[(labels[candidates] == c)].tolist() if len(candidates) > 0 else []
                if len(cand_c) == 0:
                    continue
                take = min(per_class, len(cand_c))
                sampled = rng.sample(cand_c, take)
                query_indices.extend(sampled)

            # fill remaining slots randomly from candidates
            query_indices = list(dict.fromkeys(query_indices))  # unique preserving order
            if len(query_indices) < num_query:
                remaining = list(set(candidates.tolist()) - set(query_indices))
                need = num_query - len(query_indices)
                if len(remaining) > 0:
                    add = rng.sample(remaining, min(need, len(remaining)))
                    query_indices.extend(add)

            # finalize mask
            query_indices = query_indices[:num_query]
            query_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
            if len(query_indices) > 0:
                query_mask[torch.tensor(query_indices, device=self.device, dtype=torch.long)] = True

            # Ensure synthetic graph/features exist (fallback to real graph)
            if not hasattr(self, 'synthetic_graph') or self.synthetic_graph is None:
                # For extraction evaluation we query on the real graph/features
                self.synthetic_graph = self.graph
                self.synthetic_features = self.features

            # Query target for pseudolabels on the selected nodes (timed)
            self.model.eval()
            t_target = time.time()
            with torch.no_grad():
                logits_target_all = self._forward(self.model, self.graph, self.features)
                preds_target = torch.argmax(logits_target_all, dim=1)
            # record target inference time for the full pass
            metric_comp.update(inference_target_time=(time.time() - t_target))

            # Train a surrogate to mimic the victim using only queried node logits
            surrogate = GCN(self.num_features, self.num_classes).to(self.device)
            opt = torch.optim.Adam(surrogate.parameters(), lr=surrogate_lr, weight_decay=5e-4)
            surrogate.train()
            train_start = time.time()

            for epoch in range(int(epochs_surrogate)):
                surrogate.train()
                opt.zero_grad()

                # Victim logits on the full graph but we'll only use queried nodes for loss
                t_q = time.time()
                with torch.no_grad():
                    logits_v_all = self._forward(self.model, self.graph, self.features)
                total_query_time += (time.time() - t_q)

                # Surrogate logits (same graph / features)
                logits_s_all = self._forward(surrogate, self.graph, self.features)

                if query_mask.any():
                    logits_v = logits_v_all[query_mask]
                    logits_s = logits_s_all[query_mask]
                else:
                    # fallback to full graph if no query mask
                    logits_v = logits_v_all
                    logits_s = logits_s_all

                # distillation loss only on queried nodes
                loss = F.kl_div(
                    F.log_softmax(logits_s, dim=1),
                    F.softmax(logits_v, dim=1),
                    reduction='batchmean'
                )
                loss.backward()
                opt.step()

            train_surrogate_end = time.time()

            # Evaluate surrogate on real test set
            surrogate.eval()
            t_sur = time.time()
            with torch.no_grad():
                logits_s_eval = self._forward(surrogate, self.graph, self.features)
                preds_s = torch.argmax(logits_s_eval, dim=1)
            # record surrogate inference time
            metric_comp.update(inference_surrogate_time=(time.time() - t_sur))

            # prepare metric update using test mask
            try:
                test_mask = self.test_mask.to(preds_s.device) if hasattr(self, 'test_mask') else None
            except Exception:
                test_mask = self.test_mask

            if test_mask is None:
                # fallback to evaluating on all nodes if no mask exists
                preds_test = preds_s
                labels_test = self.labels
                query_test = preds_target
            else:
                preds_test = preds_s[test_mask]
                labels_test = self.labels[test_mask]
                query_test = preds_target[test_mask]

            metric.update(preds_test, labels_test, query_test)
            metric_comp.update(
                train_surrogate_time=(train_surrogate_end - train_start),
                attack_time=(time.time() - attack_start),
                query_target_time=total_query_time,
            )
            metric_comp.end()

            perf = metric.compute()
            comp = metric_comp.compute()
            return perf, comp
        except Exception as e:
            import traceback as _tb
            tb = _tb.format_exc()
            # Always return a dict pair so the runner can continue and log the failure
            return ({'error': str(e), 'traceback': tb}, {})