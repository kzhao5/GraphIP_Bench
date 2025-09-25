import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.metrics import precision_score, recall_score, f1_score

from .base import BaseDefense
from pygip.models.nn.backbones import GCN_PyG
from pygip.utils.metrics import DefenseCompMetric, DefenseMetric
import pygip.utils.metrics as metrics


class ImperceptibleWM(BaseDefense):
    # support both PyG and DGL by converting DGL graph to a PyG-like object
    supported_api_types = {"pyg", "dgl"}

    def __init__(self, dataset, defense_ratio=0.1, model_path=None, clean_pretrain_epochs: int = 0):
        super().__init__(dataset, defense_ratio)
        # load data
        # Flag to indicate whether the target model has been trained
        self.model_trained = False
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        # If dataset is DGL, convert to a minimal PyG-like data object expected by this defense
        if getattr(dataset, "api_type", None) == "dgl":
            import types
            g = dataset.graph_data
            try:
                u, v = g.edges()
            except Exception:
                # fallback to CPU numpy edges
                edges = g.edges()
                u = torch.as_tensor(edges[0])
                v = torch.as_tensor(edges[1])
            edge_index = torch.stack([u, v], dim=0)

            new_data = types.SimpleNamespace()
            # copy tensors and move to device
            try:
                new_data.x = g.ndata['feat'].to(self.device)
            except Exception:
                new_data.x = torch.as_tensor(g.ndata['feat']).to(self.device)
            new_data.edge_index = edge_index.to(self.device)
            try:
                new_data.y = g.ndata['label'].to(self.device)
            except Exception:
                new_data.y = torch.as_tensor(g.ndata['label']).to(self.device)
            # masks
            try:
                new_data.train_mask = g.ndata['train_mask'].to(self.device)
            except Exception:
                new_data.train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool, device=self.device)
            try:
                new_data.val_mask = g.ndata['val_mask'].to(self.device)
            except Exception:
                new_data.val_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool, device=self.device)
            try:
                new_data.test_mask = g.ndata['test_mask'].to(self.device)
            except Exception:
                new_data.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool, device=self.device)
            new_data.num_nodes = dataset.num_nodes
            self.graph_data = new_data
        else:
            self.graph_data = dataset.graph_data.to(self.device)
        self.defense_ratio = defense_ratio
        self.num_triggers = int(dataset.num_nodes * defense_ratio)
        self.model_path = model_path

        self.owner_id = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9], dtype=torch.float32, device=self.graph_data.x.device)

        in_feats = dataset.num_features
        num_classes = dataset.num_classes

        self.generator = TriggerGenerator(in_feats, 64, self.owner_id).to(self.device)
        self.model = GCN_PyG(in_feats, 128, num_classes).to(self.device)
        # optional clean pretrain epochs (counts as target training time)
        self.clean_pretrain_epochs = int(clean_pretrain_epochs)

    def defend(self):
        """
        Execute the imperceptible watermark defense.
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================Imperceptible Watermark====================")

        # If model wasn't trained yet, train it
        if not self.model_trained:
            self.train_target_model(metric_comp)

        # Evaluate the watermarked model
        trigger_data = generate_trigger_graph(self.graph_data, self.generator, self.model, self.num_triggers)
        preds = self.evaluate_model(trigger_data)
        inference_s = time.time()
        wm_preds = self.verify_watermark(trigger_data)
        inference_e = time.time()

        # metric
        metric = DefenseMetric()
        metric.update(preds, trigger_data.y[trigger_data.original_test_mask])
        wm_true = trigger_data.y[trigger_data.trigger_nodes]
        metric.update_wm(wm_preds, wm_true)
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        res_comp = metric_comp.compute()

        return res, res_comp

    def train_target_model(self, metric_comp: DefenseCompMetric):
        """Train the model with optional clean pretraining + bi-level watermark optimization.

        clean_pretrain_epochs > 0 => split timing; else all counted as defense phase.
        """
        overall_start = time.time()
        clean_time = 0.0
        if self.clean_pretrain_epochs > 0:
            pre_s = time.time()
            self._clean_pretrain(self.clean_pretrain_epochs)
            clean_time = time.time() - pre_s
        defense_s = time.time()
        pyg_data = self.graph_data
        bi_level_optimization(self.model, self.generator, pyg_data, self.num_triggers)
        defense_time = time.time() - defense_s
        self.model_trained = True
        metric_comp.update(train_target_time=clean_time, train_defense_time=defense_time, defense_time=clean_time + defense_time)
        try:
            metrics.LAST_DEFENSE_TARGET_TRAIN_TIME = float(clean_time)
            metrics.LAST_DEFENSE_TRAIN_TIME = float(defense_time)
        except Exception:
            pass
        return self.model

    def _clean_pretrain(self, epochs: int):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        data = self.graph_data
        for ep in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask]) if data.train_mask.sum() > 0 else torch.tensor(0.0, device=out.device)
            loss.backward()
            optimizer.step()

    def evaluate_model(self, trigger_data):
        """Evaluate model performance on downstream task"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(trigger_data.x, trigger_data.edge_index)
            preds = out[trigger_data.original_test_mask].argmax(dim=1).cpu()
        return preds

    def verify_watermark(self, trigger_data):
        """Verify watermark success rate"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(trigger_data.x, trigger_data.edge_index)
            wm_preds = out[trigger_data.trigger_nodes].argmax(dim=1).cpu()
        return wm_preds

    def _load_model(self):
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path))

    def _train_target_model(self):
        # optional if you split training from watermarking
        pass

    def _train_defense_model(self):
        return self.model

    def _train_surrogate_model(self):
        pass


class TriggerGenerator(nn.Module):
    def __init__(self, in_channels, hidden_channels, owner_id):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, in_channels)
        self.owner_id = owner_id

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = torch.sigmoid(self.conv2(x, edge_index))
        out = x.clone()
        out[:, -5:] = self.owner_id
        return out


def generate_trigger_graph(data, generator, target_model, num_triggers=50):
    with torch.no_grad():
        probs = F.softmax(target_model(data.x, data.edge_index), dim=1)

    selected_nodes = []
    for class_idx in range(probs.size(1)):
        class_nodes = torch.where(data.y == class_idx)[0]
        if len(class_nodes) > 0:
            selected_nodes.append(class_nodes[probs[class_nodes, class_idx].argmax()].item())

    trigger_features = generator(data.x, data.edge_index)
    trigger_nodes = list(range(data.num_nodes, data.num_nodes + num_triggers))
    total_nodes = data.num_nodes + num_triggers

    # Create new dense adjacency matrix
    adj = to_dense_adj(data.edge_index)[0]
    new_adj = torch.zeros((total_nodes, total_nodes), device=adj.device)
    new_adj[:adj.size(0), :adj.size(1)] = adj

    # Connect trigger nodes to selected nodes
    for i, trigger in enumerate(trigger_nodes):
        for node in selected_nodes:
            new_adj[node, trigger] = 1
            new_adj[trigger, node] = 1

    new_data = copy.deepcopy(data)
    new_data.x = torch.cat([data.x, trigger_features[:num_triggers]], dim=0)
    new_data.edge_index = dense_to_sparse(new_adj)[0]
    new_data.y = torch.cat([
        data.y,
        torch.zeros(num_triggers, dtype=torch.long, device=data.y.device)
    ])

    new_data.train_mask = torch.cat([
        data.train_mask,
        torch.zeros(num_triggers, dtype=torch.bool, device=data.x.device)
    ])
    new_data.val_mask = torch.cat([
        data.val_mask,
        torch.zeros(num_triggers, dtype=torch.bool, device=data.x.device)
    ])
    new_data.test_mask = torch.cat([
        data.test_mask,
        torch.zeros(num_triggers, dtype=torch.bool, device=data.x.device)
    ])

    # Preserve original test mask as a mask over the expanded node set: pad with False for trigger nodes
    orig_test = data.test_mask
    if orig_test.dim() == 1 and orig_test.numel() == data.num_nodes:
        padded = torch.cat([orig_test.to(device=orig_test.device), torch.zeros(num_triggers, dtype=torch.bool, device=orig_test.device)])
    else:
        # fallback: create boolean mask of length total_nodes with first data.num_nodes entries from data.test_mask
        padded = torch.zeros(total_nodes, dtype=torch.bool, device=new_data.x.device)
        try:
            padded[:data.num_nodes] = data.test_mask.to(device=new_data.x.device)
        except Exception:
            padded[:data.num_nodes] = torch.as_tensor(data.test_mask, device=new_data.x.device)
    new_data.original_test_mask = padded

    # Add trigger info
    new_data.trigger_nodes = trigger_nodes
    new_data.selected_nodes = selected_nodes
    new_data.trigger_mask = torch.zeros(total_nodes, dtype=torch.bool, device=data.x.device)
    new_data.trigger_mask[trigger_nodes] = True

    return new_data


def bi_level_optimization(target_model, generator, data, num_triggers, epochs=100, inner_steps=5):
    optimizer_model = torch.optim.Adam(target_model.parameters(), lr=0.01)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(epochs)):
        for _ in range(inner_steps):
            optimizer_model.zero_grad()
            trigger_data = generate_trigger_graph(data, generator, target_model, num_triggers)

            out_clean = target_model(data.x, data.edge_index)
            out_trigger = target_model(trigger_data.x, trigger_data.edge_index)

            clean_loss = criterion(out_clean[data.train_mask], data.y[data.train_mask])
            trigger_loss = criterion(out_trigger[trigger_data.trigger_mask],
                                     trigger_data.y[trigger_data.trigger_mask])

            total_loss = clean_loss + trigger_loss
            total_loss.backward()
            optimizer_model.step()

        optimizer_gen.zero_grad()
        trigger_data = generate_trigger_graph(data, generator, target_model, num_triggers)

        orig_features = data.x[trigger_data.selected_nodes]
        trigger_features = trigger_data.x[trigger_data.trigger_nodes]
        sim_loss = -F.cosine_similarity(orig_features.unsqueeze(1),
                                        trigger_features.unsqueeze(0), dim=-1).mean()

        out = target_model(trigger_data.x, trigger_data.edge_index)
        trigger_loss = criterion(out[trigger_data.trigger_mask],
                                 trigger_data.y[trigger_data.trigger_mask])

        owner_loss = F.binary_cross_entropy(
            trigger_data.x[trigger_data.trigger_nodes, -5:],
            generator.owner_id.expand(len(trigger_data.trigger_nodes), 5)
        )

        total_gen_loss = 0.4 * sim_loss + 0.4 * trigger_loss + 0.2 * owner_loss
        total_gen_loss.backward()
        optimizer_gen.step()


def calculate_metrics(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        true = data.y

        # Handle both original and watermarked data cases
        if hasattr(data, 'original_test_mask'):
            test_mask = data.original_test_mask
            if test_mask.size(0) < pred.size(0):
                pad_len = pred.size(0) - test_mask.size(0)
                test_mask = torch.cat([test_mask, torch.zeros(pad_len, dtype=torch.bool, device=test_mask.device)])
        else:
            test_mask = data.test_mask

        metrics = {
            'accuracy': (pred[test_mask] == true[test_mask]).float().mean().item(),
            'precision': precision_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'recall': recall_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'f1': f1_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'wm_accuracy': None
        }

        if hasattr(data, 'trigger_nodes'):
            wm_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=data.x.device)
            wm_mask[data.trigger_nodes] = True
            wm_pred = pred[wm_mask]
            wm_true = true[wm_mask]
            metrics['wm_accuracy'] = (wm_pred == wm_true).float().mean().item()

        return metrics
