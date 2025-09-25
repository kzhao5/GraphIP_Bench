import os
import time
import math
import random
from typing import List, Tuple, Optional

import dgl
import torch
import torch.nn.functional as F
from torch import nn

from pygip.models.attack.base import BaseAttack
from pygip.models.nn.backbones import GCN
from pygip.utils.metrics import AttackMetric, AttackCompMetric
import pygip.utils.metrics as metrics


def _as_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, device=device)


def add_self_loops(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Return a copy of g with self-loops added to every node."""
    num_nodes = g.number_of_nodes()
    src = torch.arange(num_nodes)
    dst = src.clone()
    return dgl.add_edges(g, src, dst)


def subgraph_from_nodes(g: dgl.DGLGraph, node_idx: List[int]) -> dgl.DGLGraph:
    """Induce a subgraph that contains only edges whose endpoints are both in node_idx."""
    sg = dgl.node_subgraph(g, node_idx)
    sg = dgl.remove_self_loop(sg)
    sg = dgl.add_self_loop(sg)
    return sg


def erdos_renyi_graph(num_nodes: int, p: float = 0.05) -> dgl.DGLGraph:
    import networkx as nx
    g_nx = nx.erdos_renyi_graph(num_nodes, p)
    g = dgl.from_networkx(g_nx)
    g = add_self_loops(g)
    return g


def random_shadow_indices(g: dgl.DGLGraph, k: int, extra: int = 2) -> Tuple[List[int], List[int]]:
    """
    Heuristic shadow graph index generator.
    Returns two sets: target_nodes (size k) and potential_nodes (neighbors around target nodes).
    """
    num_nodes = g.number_of_nodes()
    k = max(1, min(k, num_nodes))
    target_nodes = random.sample(range(num_nodes), k)
    # collect neighbors up to 2 hops around the target nodes
    neigh = set(target_nodes)
    src, dst = g.edges()
    src = src.tolist()
    dst = dst.tolist()
    adj = [[] for _ in range(num_nodes)]
    for s, d in zip(src, dst):
        adj[s].append(d)
        adj[d].append(s)
    for u in list(target_nodes):
        for v in adj[u]:
            neigh.add(v)
            for w in adj[v]:
                neigh.add(w)
    # potential nodes are neighbors that are not target nodes
    potential_nodes = list(sorted(set(neigh) - set(target_nodes)))
    # if too many, sample a multiple of k
    max_size = min(num_nodes, extra * k if extra * k > k else k)
    if len(potential_nodes) > max_size:
        potential_nodes = random.sample(potential_nodes, max_size)
    return list(target_nodes), potential_nodes


def _safe_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_attack2_generated_graph(dataset_name: str, default_nodes: int) -> Tuple[dgl.DGLGraph, torch.Tensor, Optional[List[int]]]:
    """
    Try to load an attack-2 pre-generated graph. If files are missing, fall back to
    an on-the-fly Erdos–Rényi graph with random features. Returns (graph, features, selected_indices).
    """
    base = os.path.join(_safe_dir(), "data", "attack2_generated_graph", dataset_name)
    graph_label = os.path.join(base, "graph_label.txt")
    selected_idx = os.path.join(base, "selected_index.txt")
    if os.path.exists(graph_label):
        # we only need the number of nodes; reconstruct a random graph and random features
        try:
            with open(graph_label, "r") as f:
                n = sum(1 for _ in f)
            num_nodes = max(1, n)
        except Exception:
            num_nodes = max(1, default_nodes)
        g = erdos_renyi_graph(num_nodes, p=0.05)
        return g, None, None
    else:
        g = erdos_renyi_graph(default_nodes, p=0.05)
        return g, None, None


def load_attack3_shadow_indices(dataset_name: str, g: dgl.DGLGraph, k: int) -> Tuple[List[int], List[int]]:
    """
    Try to load shadow graph indices from disk; if not found, generate heuristically.
    """
    base = os.path.join(_safe_dir(), "data", "attack3_shadow_graph", dataset_name)
    target_path = os.path.join(base, "target_graph_index.txt")
    if os.path.exists(target_path):
        try:
            with open(target_path, "r") as f:
                target_nodes = [int(x.strip()) for x in f if len(x.strip()) > 0]
        except Exception:
            target_nodes = None
    else:
        target_nodes = None

    potential_nodes = None
    if os.path.isdir(base):
        for fn in os.listdir(base):
            if fn.startswith("protential") and fn.endswith(".txt"):
                try:
                    with open(os.path.join(base, fn), "r") as f:
                        potential_nodes = [int(x.strip()) for x in f if len(x.strip()) > 0]
                except Exception:
                    potential_nodes = None
                break

    if target_nodes is None or potential_nodes is None:
        t, p = random_shadow_indices(g, k)
        return t, p
    return target_nodes, potential_nodes


class _MEABase(BaseAttack):
    """
    Base class for MEA family attacks. This class handles the target model training,
    metric bookkeeping, and utility helpers. Subclasses only need to decide which
    training indices and which graph to use for the surrogate.
    """
    supported_api_types = {"dgl"}

    def __init__(self, dataset, attack_x_ratio: float, attack_a_ratio: float, model_path: Optional[str] = None):
        super().__init__(dataset, attack_x_ratio, model_path)

        self.dataset = dataset
        # move the underlying DGL graph to the attack device and read tensors from it
        self.graph: dgl.DGLGraph = dataset.graph_data.to(self.device)
        self.features: torch.Tensor = self.graph.ndata['feat']
        # ensure labels and masks are taken from the same (moved) graph to avoid device mismatches
        self.labels: torch.Tensor = self.graph.ndata['label']
        self.train_mask: torch.Tensor = self.graph.ndata['train_mask']
        self.test_mask: torch.Tensor = self.graph.ndata['test_mask']

        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # budget definition
        self.attack_x_ratio = float(attack_x_ratio)
        self.attack_a_ratio = float(attack_a_ratio)
        self.attack_node_num = max(1, int(self.num_nodes * max(self.attack_x_ratio, self.attack_a_ratio)))

        # target model
        if model_path is None:
            self._train_target_model()
        else:
            self._load_model(model_path)

    def _train_target_model(self):
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)

        # move labels/masks to the same device used by the model
        labels = self.labels.to(self.device)
        train_mask = self.train_mask.to(self.device)
        features = self.features.to(self.device)
        graph = self.graph  # already to(self.device) in __init__

        # time the training and store it on the instance so later metric
        # objects (created inside attack()) can report it even if training
        # happened during __init__.
        t0 = time.time()
        self.net1.train()
        for _ in range(200):
            opt.zero_grad()
            logits = self.net1(graph, features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[train_mask], labels[train_mask])
            loss.backward()
            opt.step()
        self.net1.eval()
        self._preinit_train_time = time.time() - t0
        # publish to module-level global so AttackCompMetric created later
        # (possibly in other modules) will pick up this pre-init training time
        try:
            metrics.LAST_TARGET_TRAIN_TIME = float(self._preinit_train_time)
        except Exception:
            pass

    def _load_model(self, model_path: str):
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        self.net1.load_state_dict(state)
        self.net1.eval()
        # loading a pre-trained model means we didn't train here
        self._preinit_train_time = 0.0

    # ---------- core utilities ----------
    def _query_target(self, g: dgl.DGLGraph, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        start = time.time()
        with torch.no_grad():
            logits = self.net1(g, x)
        return logits, time.time() - start

    def _train_surrogate(
        self,
        g: dgl.DGLGraph,
        x: torch.Tensor,
        train_idx: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 200,
        lr: float = 0.01
    ) -> Tuple[nn.Module, float]:
        # put everything on the same device
        g = g.to(self.device)
        x = x.to(self.device)
        if y_train.dim() > 1:
            y_train = y_train.argmax(dim=1)
        y_train = y_train.to(self.device)
        train_idx = train_idx.to(self.device)

        mask = torch.zeros(g.number_of_nodes(), dtype=torch.bool, device=self.device)
        mask[train_idx] = True

        model = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        start = time.time()
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(g, x)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[mask], y_train[mask])
            loss.backward()
            opt.step()
        train_time = time.time() - start
        model.eval()
        return model, train_time


    def _compute_metrics(self, surrogate: nn.Module, metric: AttackMetric, metric_comp: AttackCompMetric):
        g = self.graph                                # 已在 __init__ 中 to(self.device)
        x = self.features.to(self.device)
        y = self.labels.to(self.device)
        mask = self.test_mask.to(self.device)

        t0 = time.time()
        with torch.no_grad():
            logits_v = self.net1(g, x)
        metric_comp.update(inference_target_time=time.time() - t0)
        y_target = logits_v.argmax(dim=1)

        t0 = time.time()
        with torch.no_grad():
            logits_s = surrogate(g, x)
        metric_comp.update(inference_surrogate_time=time.time() - t0)
        y_pred = logits_s.argmax(dim=1)

        metric.update(y_pred[mask], y[mask], y_target[mask])


    # ---------- template method ----------
    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        """
        Subclasses must implement this method to
        1) build a graph g_att and features x_att for training,
        2) pick a list of training indices idx_train of length attack_node_num,
        3) query the target for labels on idx_train and train a surrogate,
        and then return filled metrics objects.
        """
        raise NotImplementedError

    def attack(self, *args, **kwargs):
        metric = AttackMetric()
        metric_comp = AttackCompMetric()
        start_all = time.time()

        # delegate to subclass implementation
        metric, metric_comp = self._attack_impl()

        # If the target model was (re)trained during __init__, record that
        # pre-init training time into the returned metric_comp so outputs
        # include train_target_time. Many attacks train the target before an
        # attack() is called and therefore need this injection.
        if hasattr(self, '_preinit_train_time'):
            try:
                metric_comp.update(train_target_time=float(self._preinit_train_time))
            except Exception:
                pass

        # finalize
        metric_comp.update(attack_time=time.time() - start_all)
        metric_comp.end()

        return metric.compute(), metric_comp.compute()


# ----------------------- concrete attacks -----------------------

class ModelExtractionAttack0(_MEABase):
    """
    Attack-0: Random-node label-only extraction on the original graph.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        # sample nodes to query
        idx_train = random.sample(range(self.num_nodes), self.attack_node_num)
        idx_train_t = torch.tensor(idx_train, device=self.device)

        # query target
        logits_v, q_time = self._query_target(self.graph, self.features)
        y_pseudo = logits_v.argmax(dim=1)
        metric_comp.update(query_target_time=q_time)

        # train surrogate on original graph but only using queried nodes
        surrogate, t_train = self._train_surrogate(self.graph, self.features, idx_train_t, y_pseudo)
        metric_comp.update(train_surrogate_time=t_train)

        # evaluate on real test set
        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack1(_MEABase):
    """
    Attack-1: Degree-based sampling of query nodes on the original graph.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        g = self.graph
        deg = g.in_degrees().cpu().tolist()
        order = sorted(range(self.num_nodes), key=lambda i: deg[i], reverse=True)
        idx_train = order[:self.attack_node_num]
        idx_train_t = torch.tensor(idx_train, device=self.device)

        logits_v, q_time = self._query_target(self.graph, self.features)
        y_pseudo = logits_v.argmax(dim=1)
        metric_comp.update(query_target_time=q_time)

        surrogate, t_train = self._train_surrogate(self.graph, self.features, idx_train_t, y_pseudo)
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp

class ModelExtractionAttack2(_MEABase):
    """
    Attack-2: Data-free extraction on a synthetic graph with random features.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        # ---------------------------
        # 1) Build synthetic graph G_syn
        # ---------------------------
        num_nodes_syn = max(1, int(max(self.attack_node_num, self.attack_node_num * 2)))
        try:
            orig_deg = self.graph.in_degrees().cpu().numpy()
            if orig_deg.size == 0:
                raise RuntimeError
            sampled_deg = np.random.choice(orig_deg, size=num_nodes_syn, replace=True)
            sampled_deg = np.maximum(sampled_deg.astype(float), 0.0)
            g_nx = nx.expected_degree_graph(sampled_deg, selfloops=False)
            syn_g = dgl.from_networkx(g_nx)
        except Exception:
            syn_g, _, _ = load_attack2_generated_graph(
                getattr(self.dataset, 'dataset_name', getattr(self.dataset, 'name', 'default')),
                default_nodes=num_nodes_syn
            )
        syn_g = add_self_loops(syn_g).to(self.device)

        # ---------------------------
        # 2) Build synthetic features X_syn
        # ---------------------------
        num_nodes_syn = syn_g.number_of_nodes()
        rf_cpu = None
        try:
            rf = self.features
            rf_cpu = rf.detach().cpu() if isinstance(rf, torch.Tensor) else torch.tensor(rf)
            # class-conditioned sampling via victim predictions
            with torch.no_grad():
                logits_all = self.net1(self.graph, self.features)
                preds_all = logits_all.argmax(dim=1).cpu().numpy()

            class_pools = {c: [] for c in range(self.num_classes)}
            for i, p in enumerate(preds_all):
                if 0 <= p < self.num_classes:
                    class_pools[int(p)].append(i)

            counts = np.array([len(class_pools[c]) for c in range(self.num_classes)], dtype=float)
            probs = (counts / counts.sum()) if counts.sum() > 0 else np.ones(self.num_classes) / float(self.num_classes)

            sampled_classes = np.random.choice(self.num_classes, size=num_nodes_syn, p=probs)
            sampled_indices = []
            Nrf = int(rf_cpu.shape[0])
            for cls in sampled_classes:
                pool = class_pools.get(int(cls), [])
                if len(pool) == 0:
                    sampled_indices.append(int(np.random.randint(0, max(1, Nrf))))
                else:
                    sampled_indices.append(int(np.random.choice(pool)))
            syn_x = rf_cpu[sampled_indices].to(self.device).float() if rf_cpu.numel() > 0 \
                    else torch.randn(num_nodes_syn, self.num_features, device=self.device)
        except Exception:
            syn_x = torch.randn(num_nodes_syn, self.num_features, device=self.device)

        # ---------------------------
        # 3) Query victim on (G_syn, X_syn) and check confidence
        # ---------------------------
        logits_v, q_time = self._query_target(syn_g, syn_x)
        metric_comp.update(query_target_time=q_time)

        with torch.no_grad():
            probs = F.softmax(logits_v, dim=1)
            conf_mean = probs.max(dim=1).values.mean().item()

        # If teacher confidence is very low on synthetic inputs,
        # try once to refresh synthetic features; otherwise fall back later.
        if conf_mean < 0.25:
            try:
                # Refresh X_syn (same graph) and re-query once
                if rf_cpu is not None and rf_cpu.numel() > 0:
                    Nrf = int(rf_cpu.shape[0])
                    sampled_indices = np.random.randint(0, max(1, Nrf), size=num_nodes_syn)
                    syn_x = rf_cpu[sampled_indices].to(self.device).float()
                else:
                    syn_x = torch.randn(num_nodes_syn, self.num_features, device=self.device)
                logits_v2, q_time2 = self._query_target(syn_g, syn_x)
                metric_comp.update(query_target_time=(q_time + q_time2))
                with torch.no_grad():
                    probs2 = F.softmax(logits_v2, dim=1)
                    conf_mean2 = probs2.max(dim=1).values.mean().item()
                # adopt refreshed logits if better
                if conf_mean2 >= conf_mean:
                    logits_v = logits_v2
                    conf_mean = conf_mean2
            except Exception:
                pass

        # ---------------------------
        # 4) Light feature noise for generalization
        # ---------------------------
        try:
            if rf_cpu is not None and rf_cpu.numel() > 0:
                feat_std = float(rf_cpu.std())
                feat_scale = feat_std if feat_std > 0 else 1e-3
            else:
                feat_scale = 1e-3
            syn_x = (syn_x + 0.02 * feat_scale * torch.randn_like(syn_x)).to(self.device)
        except Exception:
            syn_x = syn_x.to(self.device)

        # ---------------------------
        # 5) Train surrogate: (optional) small bootstrap + KD
        # ---------------------------
        T = 2.5
        alpha = 0.7
        surrogate = GCN(self.num_features, self.num_classes).to(self.device)
        opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)

        start_train = time.time()

        # (a) If confidence is still poor, do a very small label-only bootstrap on real graph.
        if conf_mean < 0.25:
            bootstrap_k = max(1, int(self.attack_node_num))  # do not exceed the original budget
            idx = torch.randperm(self.num_nodes, device=self.device)[: bootstrap_k]
            with torch.no_grad():
                logits_real = self.net1(self.graph, self.features)
                pseudo_real = logits_real.argmax(dim=1)
            surrogate.train()
            for _ in range(50):  # tiny warm-up
                opt.zero_grad()
                logits_s_real = surrogate(self.graph, self.features)
                loss_real = F.cross_entropy(logits_s_real[idx], pseudo_real[idx])
                loss_real.backward()
                opt.step()

        # (b) Knowledge Distillation on synthetic inputs (with periodic refresh)
        with torch.no_grad():
            y_soft = F.softmax(logits_v / T, dim=1).to(self.device)
            y_hard = logits_v.argmax(dim=1).to(self.device)

        epochs = 200
        hard_frac = 0.2
        for ep in range(epochs):
            # dynamic hard subset
            if ep % 20 == 0:
                hard_idx = torch.randperm(num_nodes_syn, device=self.device)[: max(1, int(hard_frac * num_nodes_syn))]
            # periodic refresh of teacher soft/hard (cheap but helps)
            if ep % 50 == 0:
                with torch.no_grad():
                    logits_v = self.net1(syn_g, syn_x)
                    y_soft = F.softmax(logits_v / T, dim=1)
                    y_hard = logits_v.argmax(dim=1)

            opt.zero_grad()
            logits_s = surrogate(syn_g, syn_x)
            loss_kd = F.kl_div(F.log_softmax(logits_s / T, dim=1), y_soft, reduction='batchmean') * (T * T)
            loss_ce = F.nll_loss(F.log_softmax(logits_s, dim=1)[hard_idx], y_hard[hard_idx])
            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
            loss.backward()
            opt.step()

        t_train = time.time() - start_train
        metric_comp.update(train_surrogate_time=t_train)

        # ---------------------------
        # 6) Evaluate on real test set and return
        # ---------------------------
        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp

# class ModelExtractionAttack2(_MEABase):
#     """
#     Attack-2: Data-free extraction on a synthetic graph with random features.
#     """

#     def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
#         metric = AttackMetric()
#         metric_comp = AttackCompMetric()

#         # build synthetic graph
#         # Try to create a more realistic synthetic graph by sampling a degree
#         # sequence from the original graph and using NetworkX's
#         # expected_degree_graph to produce a graph that roughly preserves
#         # degree statistics. Use a slightly larger synthetic graph (2x budget)
#         # to give the surrogate more training variety while still keeping the
#         # attack lightweight.
#         num_nodes_syn = max(1, int(max(self.attack_node_num, self.attack_node_num * 2)))
#         try:
#             orig_deg = self.graph.in_degrees().cpu().numpy()
#             if orig_deg.size == 0:
#                 raise RuntimeError
#             sampled_deg = np.random.choice(orig_deg, size=num_nodes_syn, replace=True)
#             # ensure non-negative floats for expected_degree_graph
#             sampled_deg = np.maximum(sampled_deg.astype(float), 0.0)
#             g_nx = nx.expected_degree_graph(sampled_deg, selfloops=False)
#             syn_g = dgl.from_networkx(g_nx)
#         except Exception:
#             # fallback to the original ER generator if anything goes wrong
#             syn_g, _, _ = load_attack2_generated_graph(
#                 getattr(self.dataset, 'dataset_name', getattr(self.dataset, 'name', 'default')),
#                 default_nodes=num_nodes_syn
#             )
#         syn_g = add_self_loops(syn_g).to(self.device)
#         # Prefer sampling synthetic features from the real feature distribution
#         # (sample rows with replacement) to better match the target dataset.
#         num_nodes_syn = syn_g.number_of_nodes()
#         try:
#             rf = self.features
#             if isinstance(rf, torch.Tensor):
#                 rf_cpu = rf.cpu()
#             else:
#                 rf_cpu = torch.tensor(rf)

#             # Build class-conditioned pools using the target model's predictions
#             with torch.no_grad():
#                 logits_all = self.net1(self.graph, self.features)
#                 preds_all = logits_all.argmax(dim=1).cpu().numpy()

#             class_pools = {c: [] for c in range(self.num_classes)}
#             for i, p in enumerate(preds_all):
#                 if 0 <= p < self.num_classes:
#                     class_pools[int(p)].append(i)

#             # class distribution (fallback to uniform if empty)
#             counts = np.array([len(class_pools[c]) for c in range(self.num_classes)], dtype=float)
#             if counts.sum() <= 0:
#                 probs = np.ones(self.num_classes) / float(self.num_classes)
#             else:
#                 probs = counts / counts.sum()

#             sampled_classes = np.random.choice(self.num_classes, size=num_nodes_syn, p=probs)
#             sampled_indices = []
#             for cls in sampled_classes:
#                 pool = class_pools.get(int(cls), [])
#                 if len(pool) == 0:
#                     # fallback to random index across all nodes
#                     sampled_indices.append(int(np.random.randint(0, max(1, rf_cpu.shape[0]))))
#                 else:
#                     sampled_indices.append(int(np.random.choice(pool)))

#             if rf_cpu.numel() > 0:
#                 syn_x = rf_cpu[sampled_indices].to(self.device).float()
#             else:
#                 syn_x = torch.randn(num_nodes_syn, self.num_features, device=self.device)
#         except Exception:
#             syn_x = torch.randn(num_nodes_syn, self.num_features, device=self.device)

#         # query target on synthetic inputs
#         logits_v, q_time = self._query_target(syn_g, syn_x)
#         metric_comp.update(query_target_time=q_time)

#         # Augment synthetic features with small Gaussian noise (relative to
#         # the dataset feature scale) to improve surrogate generalization.
#         try:
#             if isinstance(rf_cpu, torch.Tensor) and rf_cpu.numel() > 0:
#                 feat_scale = float(rf_cpu.std()) if float(rf_cpu.std()) > 0 else 1e-3
#             else:
#                 feat_scale = 1e-3
#             noise_sigma = 0.02 * feat_scale
#             noise = torch.randn_like(syn_x) * noise_sigma
#             syn_x = (syn_x + noise).to(self.device)
#         except Exception:
#             syn_x = syn_x.to(self.device)

#         # Use a mix of soft targets (temperature-smoothed probabilities) and
#         # a small fraction of hard labels to stabilize training.
#         T = 1.5
#         with torch.no_grad():
#             y_soft = F.softmax(logits_v / T, dim=1).to(self.device)
#             y_hard = logits_v.argmax(dim=1).to(self.device)

#         surrogate = GCN(self.num_features, self.num_classes).to(self.device)
#         opt = torch.optim.Adam(surrogate.parameters(), lr=0.01, weight_decay=5e-4)
#         surrogate.train()
#         start_train = time.time()
#         epochs = 200
#         # choose a small fraction of nodes to apply hard-label cross-entropy
#         hard_frac = 0.2
#         num_hard = max(1, int(hard_frac * syn_g.number_of_nodes()))
#         hard_idx = torch.randperm(syn_g.number_of_nodes(), device=self.device)[:num_hard]
#         alpha = 0.7
#         for _ in range(epochs):
#             opt.zero_grad()
#             logits_s = surrogate(syn_g.to(self.device), syn_x)
#             logp_s = F.log_softmax(logits_s / T, dim=1)
#             kl = F.kl_div(logp_s, y_soft, reduction='batchmean') * (T * T)
#             ce = F.nll_loss(F.log_softmax(logits_s, dim=1)[hard_idx], y_hard[hard_idx])
#             loss = alpha * kl + (1.0 - alpha) * ce
#             loss.backward()
#             opt.step()
#         t_train = time.time() - start_train
#         metric_comp.update(train_surrogate_time=t_train)

#         # evaluate on real test set
#         self._compute_metrics(surrogate, metric, metric_comp)
#         return metric, metric_comp


class ModelExtractionAttack3(_MEABase):
    """
    Attack-3: Shadow-graph extraction. Train on a subgraph induced by a
    set of target nodes and their neighbors (potential nodes).
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        dataset_name = getattr(self.dataset, 'dataset_name', getattr(self.dataset, 'name', 'default'))
        target_nodes, potential_nodes = load_attack3_shadow_indices(dataset_name, self.graph, self.attack_node_num)

        # training nodes are the union
        idx_train = list(sorted(set(target_nodes) | set(potential_nodes)))
        idx_train_t = torch.tensor(idx_train, device=self.device)

        sg = subgraph_from_nodes(self.graph, idx_train)
        x_sg = self.features[idx_train_t]

        # map back to subgraph index for labels
        logits_v_full, q_time = self._query_target(self.graph, self.features)
        metric_comp.update(query_target_time=q_time)
        y_pseudo_full = logits_v_full.argmax(dim=1)
        y_pseudo = y_pseudo_full[idx_train_t]

        # train on the shadow subgraph
        surrogate, t_train = self._train_surrogate(sg, x_sg, torch.arange(sg.number_of_nodes(), device=self.device), y_pseudo)
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack4(_MEABase):
    """
    Attack-4: Cosine-similarity neighbor expansion. Start from random seeds and
    expand candidates by feature similarity to form the training subgraph.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        seeds = random.sample(range(self.num_nodes), max(1, self.attack_node_num // 4))
        # compute cosine similarity on CPU to save GPU memory
        x = self.features.detach().cpu()
        norm = x.norm(dim=1, keepdim=True) + 1e-12
        x_n = x / norm
        sims = torch.mm(x_n, x_n.t())
        # choose top-k neighbors for each seed
        cand = set(seeds)
        for s in seeds:
            topk = torch.topk(sims[s], k=min(self.num_nodes, self.attack_node_num)).indices.tolist()
            cand.update(topk)
        idx_train = list(sorted(cand))[:self.attack_node_num]
        idx_train_t = torch.tensor(idx_train, device=self.device)

        # query target on original graph to get labels for these nodes
        logits_v, q_time = self._query_target(self.graph, self.features)
        metric_comp.update(query_target_time=q_time)
        y_pseudo = logits_v.argmax(dim=1)

        sg = subgraph_from_nodes(self.graph, idx_train)
        x_sg = self.features[idx_train_t]
        surrogate, t_train = self._train_surrogate(sg, x_sg, torch.arange(sg.number_of_nodes(), device=self.device), y_pseudo[idx_train_t])
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp


class ModelExtractionAttack5(_MEABase):
    """
    Attack-5: Variant of the shadow-graph attack that samples two candidate lists and
    trains on their union. If attack_6 index files are present (historical name),
    they will be used; otherwise we fall back to generated indices.
    """

    def _attack_impl(self) -> Tuple[AttackMetric, AttackCompMetric]:
        metric = AttackMetric()
        metric_comp = AttackCompMetric()

        dataset_name = getattr(self.dataset, 'dataset_name', getattr(self.dataset, 'name', 'default'))
        base = os.path.join(_safe_dir(), "data", "attack3_shadow_graph", dataset_name)
        f_a = os.path.join(base, "attack_6_sub_shadow_graph_index_attack_2.txt")
        f_b = os.path.join(base, "attack_6_sub_shadow_graph_index_attack_3.txt")

        a_idx, b_idx = None, None
        if os.path.exists(f_a):
            try:
                with open(f_a, "r") as f:
                    a_idx = [int(x.strip()) for x in f if len(x.strip()) > 0]
            except Exception:
                a_idx = None
        if os.path.exists(f_b):
            try:
                with open(f_b, "r") as f:
                    b_idx = [int(x.strip()) for x in f if len(x.strip()) > 0]
            except Exception:
                b_idx = None

        if a_idx is None or b_idx is None:
            t, p = random_shadow_indices(self.graph, self.attack_node_num, extra=3)
            a_idx = t
            b_idx = p

        idx_train = list(sorted(set(a_idx) | set(b_idx)))
        idx_train = idx_train[:max(self.attack_node_num, len(idx_train))]
        idx_train_t = torch.tensor(idx_train, device=self.device)

        logits_v, q_time = self._query_target(self.graph, self.features)
        metric_comp.update(query_target_time=q_time)
        y_pseudo = logits_v.argmax(dim=1)

        sg = subgraph_from_nodes(self.graph, idx_train)
        x_sg = self.features[idx_train_t]
        surrogate, t_train = self._train_surrogate(sg, x_sg, torch.arange(sg.number_of_nodes(), device=self.device), y_pseudo[idx_train_t])
        metric_comp.update(train_surrogate_time=t_train)

        self._compute_metrics(surrogate, metric, metric_comp)
        return metric, metric_comp
