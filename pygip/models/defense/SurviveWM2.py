import random
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm
from torch_geometric.utils import to_networkx

from pygip.models.defense.base import BaseDefense

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SAGEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 10, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch, return_embedding=False):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            if i < len(self.convs) - 1:
                x = self.dropout(x)
        embedding = global_mean_pool(x, batch)
        if return_embedding:
            return embedding
        return self.classifier(embedding)


class WatermarkGenerator:
    def __init__(self, training_dataset: List[Data], num_watermark_samples: int = None):
        self.training_dataset = training_dataset
        self.num_classes = self._get_num_classes()
        self.avg_num_nodes = self._get_avg_num_nodes()
        self.feature_dim = training_dataset[0].x.size(1) if training_dataset else 32
        if num_watermark_samples is None:
            self.num_watermark_samples = max(5, int(0.05 * len(training_dataset)))
        else:
            self.num_watermark_samples = num_watermark_samples

    def algorithm_1_key_input_topology_generation(self, N_t: int, N: int = None) -> Data:
        if N is None:
            N = min(50, self.avg_num_nodes)
        x = torch.zeros(N, self.feature_dim)
        p = 0.5
        G_r = nx.erdos_renyi_graph(N_t, p)
        remaining_nodes = N - N_t
        if remaining_nodes > 0:
            training_sample = random.choice(self.training_dataset)
            G_train = to_networkx(training_sample, to_undirected=True)
            if G_train.number_of_nodes() >= remaining_nodes:
                sampled_nodes = random.sample(list(G_train.nodes()), remaining_nodes)
                G_o = G_train.subgraph(sampled_nodes).copy()
                for i, node in enumerate(sampled_nodes):
                    if node < training_sample.x.size(0):
                        x[N_t + i] = training_sample.x[node]
                    else:
                        x[N_t + i] = torch.randn(self.feature_dim)
            else:
                G_o = G_train.copy()
                for i in range(min(remaining_nodes, training_sample.x.size(0))):
                    x[N_t + i] = training_sample.x[i]
        else:
            G_o = nx.Graph()
        edges = set()
        for u, v in G_r.edges():
            if u < N and v < N:
                edges.add((min(u, v), max(u, v)))
        node_mapping = {old: new + N_t for new, old in enumerate(G_o.nodes())}
        for u_old, v_old in G_o.edges():
            u, v = node_mapping[u_old], node_mapping[v_old]
            if u < N and v < N:
                edges.add((min(u, v), max(u, v)))
        if edges:
            edge_list = []
            for u, v in edges:
                edge_list.extend([[u, v], [v, u]])
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        watermark_graph = Data(x=x, edge_index=edge_index)
        return watermark_graph

    def generate_watermark_set_with_clean_model(self, clean_model) -> List[Tuple[Data, int]]:
        watermark_pairs = []
        while len(watermark_pairs) < self.num_watermark_samples:
            N_t = random.choice([3, 4])
            watermark_graph = self.algorithm_1_key_input_topology_generation(N_t)
            clean_model.eval()
            with torch.no_grad():
                batch = torch.zeros(watermark_graph.x.size(0), dtype=torch.long)
                pred_logits = clean_model(watermark_graph.x, watermark_graph.edge_index, batch)
                clean_pred = pred_logits.argmax().item()
                probs = F.softmax(pred_logits, dim=1)
                if probs.max().item() < 0.6:
                    continue
            other_classes = [i for i in range(self.num_classes) if i != clean_pred]
            watermark_label = random.choice(other_classes) if other_classes else (clean_pred + 1) % self.num_classes
            watermark_pairs.append((watermark_graph, watermark_label))
        while len(watermark_pairs) < max(5, self.num_watermark_samples // 2):
            wg = self.algorithm_1_key_input_topology_generation(random.choice([2, 3, 4]))
            watermark_pairs.append((wg, random.randint(0, self.num_classes - 1)))
        return watermark_pairs

    def _get_num_classes(self) -> int:
        if not self.training_dataset:
            return 2
        labels = {int(d.y) for d in self.training_dataset if hasattr(d, 'y')}
        return max(labels) + 1 if labels else 2

    def _get_avg_num_nodes(self) -> int:
        if not self.training_dataset:
            return 20
        total = sum(d.x.size(0) for d in self.training_dataset)
        return total // len(self.training_dataset)


class SNNLLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.T = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N = embeddings.size(0)
        if N <= 1:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        distances = torch.cdist(embeddings, embeddings, p=2).pow(2)
        loss = 0.0
        count = 0
        for i in range(N):
            same_mask = (labels == labels[i]) & (torch.arange(N, device=labels.device) != i)
            diff_mask = torch.arange(N, device=labels.device) != i
            if same_mask.sum() == 0 or diff_mask.sum() == 0:
                continue
            numerator = torch.exp(-distances[i, same_mask] / self.T).sum()
            denominator = torch.exp(-distances[i, diff_mask] / self.T).sum()
            loss += -torch.log((numerator + 1e-8) / (denominator + 1e-8))
            count += 1
        return loss / max(count, 1) if count > 0 else torch.tensor(0.0, requires_grad=True, device=embeddings.device)


def train_clean_model(training_data: List[Data], epochs: int = 200, batch_size: int = 32,
                      num_layers: int = 3) -> SAGEModel:
    num_classes = max([int(d.y) for d in training_data]) + 1
    model = SAGEModel(input_dim=training_data[0].x.size(1), hidden_dim=160, num_classes=num_classes,
                      num_layers=num_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.num_graphs

        if epoch > 50 and epoch % 20 == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.8
    return model


def train_watermarked_model_full(
        training_data: List[Data],
        key_inputs: List[Tuple[Data, int]],
        epochs: int = 300,
        alpha: float = 0.1,
        num_layers: int = 4,
        hidden_dim: int = 160,
        dropout: float = 0.05,
        lr: float = 1e-3,
        snnl_temperature: float = 1.0,
):
    num_classes = max([int(d.y) for d in training_data] + [label for _, label in key_inputs]) + 1
    model = SAGEModel(
        input_dim=training_data[0].x.size(1),
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ce_criterion = nn.CrossEntropyLoss()
    snnl_criterion = SNNLLoss(temperature=snnl_temperature)
    batch_size = 32

    wm_graphs = [d for d, _ in key_inputs]
    wm_labels = [l for _, l in key_inputs]

    loader_clean = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    loader_wm = DataLoader([
        Data(x=g.x, edge_index=g.edge_index, y=torch.tensor([l])) for g, l in zip(wm_graphs, wm_labels)
    ], batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_count = 0, 0, 0
        for batch in loader_clean:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = ce_criterion(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            preds = out.argmax(dim=1)
            total_correct += (preds == batch.y.view(-1)).sum().item()
            total_count += batch.num_graphs

        wm_loss_total, wm_snnl_total, wm_batches = 0.0, 0.0, 0
        for batch in loader_wm:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss_ce = ce_criterion(out, batch.y.view(-1))
            batch_embeddings, batch_labels = [], []
            emb_wm = model(batch.x, batch.edge_index, batch.batch, return_embedding=True)
            batch_embeddings.append(emb_wm)
            batch_labels.extend([1] * emb_wm.size(0))
            try:
                clean_batch = next(iter(loader_clean))
                emb_clean = model(clean_batch.x, clean_batch.edge_index, clean_batch.batch, return_embedding=True)
                batch_embeddings.append(emb_clean)
                batch_labels.extend([0] * emb_clean.size(0))
            except StopIteration:
                pass
            if len(batch_embeddings) > 1:
                emb_t = torch.cat(batch_embeddings, dim=0)
                lbl_t = torch.tensor(batch_labels, dtype=torch.long, device=emb_t.device)
                snnl_loss = snnl_criterion(emb_t, lbl_t)
            else:
                snnl_loss = torch.tensor(0.0, device=out.device)
            loss = loss_ce + alpha * snnl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            wm_loss_total += loss_ce.item()
            wm_snnl_total += snnl_loss.item()
            wm_batches += 1

        if epoch > 50 and epoch % 20 == 0:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.8

        if epoch % 20 == 0:
            avg_clean_loss = total_loss / max(total_count, 1)
            avg_wm_loss = wm_loss_total / max(wm_batches, 1)
            avg_wm_snnl = wm_snnl_total / max(wm_batches, 1)
            model.eval()
            c_corr, wm_corr = 0, 0
            with torch.no_grad():
                for d in training_data[:20]:
                    b = torch.zeros(d.x.size(0), dtype=torch.long)
                    if model(d.x, d.edge_index, b).argmax() == int(d.y):
                        c_corr += 1
                for x_exp, lbl in key_inputs[:10]:
                    b = torch.zeros(x_exp.x.size(0), dtype=torch.long)
                    if model(x_exp.x, x_exp.edge_index, b).argmax() == lbl:
                        wm_corr += 1
            clean_acc = c_corr / 20
            wm_acc = wm_corr / min(10, len(key_inputs))
            model.train()
    return model


def evaluate_watermark_effectiveness(model: SAGEModel, key_inputs: List[Tuple[Data, int]]) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, expected_label in key_inputs:
            batch = torch.zeros(x.x.size(0), dtype=torch.long)
            pred = model(x.x, x.edge_index, batch).argmax(1).item()
            if pred == expected_label:
                correct += 1
    return correct / len(key_inputs) if key_inputs else 0.0


def evaluate_clean_accuracy(model: SAGEModel, test_data: List[Data], batch_size=32) -> float:
    if not test_data:
        return 0.0
    loader = DataLoader(test_data, batch_size=batch_size)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.num_graphs
    return correct / total if total > 0 else 0.0


class KeyInputOptimizer:
    def __init__(self, training_dataset: List[Data], key_inputs: List[Tuple[Data, int]], T_opt: int = 20):
        self.training_dataset = training_dataset
        self.key_inputs = key_inputs
        self.feature_dim = training_dataset[0].x.size(1)
        self.num_classes = max([int(d.y) for d in training_dataset] + [label for _, label in key_inputs]) + 1
        self.T_opt = T_opt

    def optimize(self):
        class MTopo(nn.Module):
            def __init__(self, N_t):
                super().__init__()
                self.N_t = N_t
                self.net = nn.Sequential(
                    nn.Linear(N_t * N_t, N_t * N_t),
                    nn.ReLU(),
                    nn.Linear(N_t * N_t, N_t * N_t),
                    nn.Sigmoid()
                )

            def forward(self, A):
                x = A.flatten().unsqueeze(0)
                out = self.net(x).reshape(self.N_t, self.N_t)
                return out

        class MFeat(nn.Module):
            def __init__(self, feat_dim, N_t):
                super().__init__()
                self.N_t = N_t
                self.feat_dim = feat_dim
                self.net = nn.Sequential(
                    nn.Linear(N_t * feat_dim, N_t * feat_dim),
                    nn.ReLU(),
                    nn.Linear(N_t * feat_dim, N_t * feat_dim)
                )

            def forward(self, F):
                x = F.flatten().unsqueeze(0)
                out = self.net(x).reshape(self.N_t, self.feat_dim)
                return out

        optimized_pairs = []
        for orig_data, label in self.key_inputs:
            N = orig_data.x.size(0)
            N_t = min(4, N)
            trigger_nodes = torch.arange(N_t)
            rest_nodes = torch.arange(N_t, N)

            # Build trigger adjacency
            A_trig = torch.zeros(N_t, N_t)
            edge_set = set(tuple(edge) for edge in orig_data.edge_index.t().tolist())
            for i, u in enumerate(trigger_nodes):
                for j, v in enumerate(trigger_nodes):
                    if (u.item(), v.item()) in edge_set:
                        A_trig[i, j] = 1
            F_trig = orig_data.x[:N_t].clone()

            m_topo = MTopo(N_t)
            m_feat = MFeat(self.feature_dim, N_t)
            opt = torch.optim.Adam(list(m_topo.parameters()) + list(m_feat.parameters()), lr=1e-2)

            for step in range(self.T_opt):
                opt.zero_grad()
                A_new = m_topo(A_trig)
                F_new = m_feat(F_trig)
                A_bin = (A_new > 0.5).float()

                # Construct the new key input graph
                new_x = orig_data.x.clone()
                new_x[:N_t] = F_new.detach().squeeze()
                edges = []
                for i in range(N_t):
                    for j in range(N_t):
                        if A_bin[i, j] > 0.5:
                            edges.append([i, j])
                for u, v in orig_data.edge_index.t().tolist():
                    if u >= N_t and v >= N_t:
                        edges.append([u, v])
                    elif (u >= N_t and v < N_t) or (u < N_t and v >= N_t):
                        edges.append([u, v])
                if edges:
                    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                data_opt = Data(x=new_x, edge_index=edge_index)

                # Train temp SAGE model on this single key input (with train data)
                temp_model = SAGEModel(self.feature_dim, hidden_dim=64, num_classes=self.num_classes, num_layers=3)
                temp_opt = torch.optim.Adam(temp_model.parameters(), lr=1e-2)
                criterion = nn.CrossEntropyLoss()
                batch = torch.zeros(data_opt.x.size(0), dtype=torch.long)
                for _ in range(2):
                    rand_data = random.choice(self.training_dataset)
                    temp_opt.zero_grad()
                    out1 = temp_model(data_opt.x, data_opt.edge_index, batch)
                    loss1 = criterion(out1, torch.tensor([label]))
                    batch_rand = torch.zeros(rand_data.x.size(0), dtype=torch.long)
                    out2 = temp_model(rand_data.x, rand_data.edge_index, batch_rand)
                    loss2 = criterion(out2, rand_data.y.view(-1))
                    loss = loss1 + loss2
                    loss.backward()
                    temp_opt.step()

                with torch.no_grad():
                    pred = temp_model(data_opt.x, data_opt.edge_index, batch)
                    ce_loss = criterion(pred, torch.tensor([label]))
                score = -ce_loss
                score = score.requires_grad_()
                score.backward()
                opt.step()

            optimized_pairs.append((data_opt, label))
        return optimized_pairs


class SurviveWM2(BaseDefense):
    def __init__(
            self,
            dataset,
            attack_node_fraction,
            model_path=None,
            alpha=0.1,
            num_layers=4,
            clean_epochs=200,
            wm_epochs=200,
            **kwargs
    ):
        super().__init__(dataset, attack_node_fraction)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.num_layers = num_layers
        self.clean_epochs = clean_epochs
        self.wm_epochs = wm_epochs

        # --- Data extraction ---
        self.train_data = getattr(dataset, 'train_data', None)
        self.test_data = getattr(dataset, 'test_data', None)

        if not (isinstance(self.train_data, list) and isinstance(self.test_data, list)):
            raise ValueError(
                "This defense only supports graph classification datasets (e.g., ENZYMES). Node-level datasets are not supported.")

        self.model_path = model_path

    def defend(self):
        """
        Main defense workflow: 
        1. Train a target model (clean)
        2. (optional) Simulate attack on target model (if implemented)
        3. Train defense (watermarked) model
        4. Evaluate defense and print detailed metrics
        Returns
        -------
        dict
            Dictionary containing performance metrics
        """
        print("=" * 60)
        print("Step 1: Train clean (victim) model")
        print("-" * 60)
        target_model = self._train_target_model()
        baseline_acc = evaluate_clean_accuracy(target_model, self.test_data)
        print(f"Test accuracy on clean (victim) model: {baseline_acc:.4f}")

        print("\nStep 2: Generate and optimize watermark key inputs")
        print("-" * 60)
        wm_gen = getattr(self, 'wm_gen', None)
        if wm_gen is None:
            self.wm_gen = WatermarkGenerator(self.train_data,
                                             num_watermark_samples=int(len(self.train_data) * self.alpha))
        key_pairs = self.wm_gen.generate_watermark_set_with_clean_model(target_model)
        optimizer = KeyInputOptimizer(self.train_data, key_pairs, T_opt=20)
        key_pairs_optimized = optimizer.optimize()
        print(f"Generated and optimized {len(key_pairs_optimized)} watermark key inputs")

        print("\nStep 3: Train defense (watermarked) model")
        print("-" * 60)
        defense_model = train_watermarked_model_full(
            self.train_data, key_pairs_optimized,
            epochs=self.wm_epochs, alpha=self.alpha, num_layers=self.num_layers
        )
        defense_acc = evaluate_clean_accuracy(defense_model, self.test_data)
        print(f"Test accuracy on defense (watermarked) model: {defense_acc:.4f}")

        print("\nStep 4: Evaluate watermark effectiveness")
        print("-" * 60)
        watermark_success = evaluate_watermark_effectiveness(defense_model, key_pairs_optimized)
        print(f"Watermark detection rate (defense model): {watermark_success:.4f}")

        acc_degradation = baseline_acc - defense_acc

        print("\nPerformance metrics:")
        print("-" * 60)
        print(f"{'Metric':<36} {'Value'}")
        print("-" * 60)
        print(f"{'Test acc. (clean model)':<36} {baseline_acc:.4f}")
        print(f"{'Test acc. (defense/watermarked)':<36} {defense_acc:.4f}")
        print(f"{'Accuracy degradation':<36} {acc_degradation:.4f}")
        print(f"{'Watermark detection (defense)':<36} {watermark_success:.4f}")
        print("-" * 60)

        results = {
            "baseline_accuracy": baseline_acc,
            "defense_accuracy": defense_acc,
            "watermark_effectiveness": watermark_success,
            "accuracy_degradation": acc_degradation,
        }
        return results

    def _load_model(self):
        if not self.model_path:
            raise ValueError("No model_path provided.")

    def _train_target_model(self):
        print("[OptimizedWatermarkDefense] Training clean (victim) model...")
        model = train_clean_model(self.train_data, epochs=self.clean_epochs, num_layers=self.num_layers)
        self.net1 = model
        self.wm_gen = WatermarkGenerator(self.train_data, num_watermark_samples=int(len(self.train_data) * self.alpha))
        return model

    def _train_defense_model(self, clean_model=None):
        print("[OptimizedWatermarkDefense] Generating and optimizing watermark key inputs...")
        if not hasattr(self, 'wm_gen'):
            self.wm_gen = WatermarkGenerator(self.train_data,
                                             num_watermark_samples=int(len(self.train_data) * self.alpha))
        key_pairs = self.wm_gen.generate_watermark_set_with_clean_model(clean_model or self.net1)
        optimizer = KeyInputOptimizer(self.train_data, key_pairs, T_opt=20)
        key_pairs_optimized = optimizer.optimize()
        print("[OptimizedWatermarkDefense] Training watermarked model...")
        model = train_watermarked_model_full(
            self.train_data, key_pairs_optimized,
            epochs=self.wm_epochs, alpha=self.alpha, num_layers=self.num_layers
        )
        self.net2 = model
        self.key_pairs_optimized = key_pairs_optimized
        return model, key_pairs_optimized

    def _train_surrogate_model(self):
        pass
