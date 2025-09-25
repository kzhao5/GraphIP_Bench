import random
from time import time

import torch
import torch.nn.functional as F

from pygip.models.defense.base import BaseDefense
from pygip.models.nn import GCN
from pygip.utils.metrics import DefenseMetric, DefenseCompMetric
import pygip.utils.metrics as metrics


class BackdoorWM(BaseDefense):
    supported_api_types = {"dgl"}

    def __init__(self, dataset, attack_node_fraction: float = 0.01, model_path=None,
                 trigger_rate=0.01, l=20, target_label=0, trigger_density=None,
                 alpha: float = 0.3, trigger_feat_val: float = 0.99, epochs: int = 200, pretrain_epochs: int = 100):
        # allow configs to pass 'trigger_density' (legacy name) -> map to attack_node_fraction
        if trigger_density is not None:
            attack_node_fraction = trigger_density
        super().__init__(dataset, attack_node_fraction)
        # load data
        self.dataset = dataset
        self.graph_dataset = dataset.graph_dataset
        
        # Try to move the DGL graph to the configured device. If that fails (e.g. DGL/CUDA
        # mismatch), fall back to keeping the original graph and copy ndata tensors manually
        # to the chosen device so training/evaluation can proceed.
        try:
            # Prefer using the graph's .to(...) which is efficient when device is valid
            self.graph_data = dataset.graph_data.to(device=self.device)

            # Extract tensors (these will already be on self.device when .to succeeded)
            self.features = self.graph_data.ndata['feat']
            self.labels = self.graph_data.ndata['label']
            self.train_mask = self.graph_data.ndata['train_mask']
            self.test_mask = self.graph_data.ndata['test_mask']
        except Exception as e:
            # Fallback: don't move the DGL graph (avoids DGL attempting CUDA ops),
            # but copy the per-node tensors to the chosen device.
            print(f"Warning: failed to move DGL graph to {self.device} ({e}). Falling back to copying ndata tensors.")
            self.graph_data = dataset.graph_data
            # copy ndata fields to device with safe conversions
            try:
                self.features = torch.as_tensor(self.graph_data.ndata['feat']).to(self.device)
            except Exception:
                self.features = self.graph_data.ndata['feat'] if isinstance(self.graph_data.ndata['feat'], torch.Tensor) else torch.tensor(self.graph_data.ndata['feat'], device=self.device)

            try:
                self.labels = torch.as_tensor(self.graph_data.ndata['label']).to(self.device)
            except Exception:
                self.labels = self.graph_data.ndata['label'] if isinstance(self.graph_data.ndata['label'], torch.Tensor) else torch.tensor(self.graph_data.ndata['label'], device=self.device)

            # masks may be missing; provide safe defaults
            try:
                self.train_mask = torch.as_tensor(self.graph_data.ndata['train_mask']).to(self.device)
            except Exception:
                self.train_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool, device=self.device)

            try:
                self.test_mask = torch.as_tensor(self.graph_data.ndata['test_mask']).to(self.device)
            except Exception:
                self.test_mask = torch.zeros(dataset.num_nodes, dtype=torch.bool, device=self.device)

        self.model_path = model_path

        # load meta data
        self.num_nodes = dataset.num_nodes
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        # wm params
        # Support legacy config key 'trigger_density' and prefer it as the effective trigger rate
        # If a user passed trigger_density in configs, use it as trigger_rate; otherwise use trigger_rate arg
        if trigger_density is not None:
            self.trigger_rate = float(trigger_density)
        else:
            # fallback to provided trigger_rate arg (default 0.01)
            self.trigger_rate = float(trigger_rate)
        self.l = l
        self.target_label = target_label
        # backdoor training control: weight for backdoor loss, feature value used for trigger, and epochs
        self.alpha = float(alpha)
        self.trigger_feat_val = float(trigger_feat_val)
        self.epochs = int(epochs)
        # number of clean-only pretraining epochs to protect task accuracy
        self.pretrain_epochs = int(pretrain_epochs)

    def _load_model(self):
        """
        Load a pre-trained model.
        """
        assert self.model_path, "self.model_path should be defined"

        # Create the model
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)

        # Load the saved state dict
        self.net1.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # Set to evaluation mode
        self.net1.eval()

    def inject_backdoor_trigger(self, data, trigger_rate=None, trigger_feat_val=0.99, l=None, target_label=None):
        """Feature-based Trigger Injection"""
        if trigger_rate is None:
            trigger_rate = self.trigger_rate
        if l is None:
            l = self.l
        if target_label is None:
            target_label = self.target_label

        num_nodes = data.shape[0]
        num_feats = data.shape[1]
        # ensure l does not exceed the number of features
        l = min(l, num_feats)

        # compute number of trigger nodes; when possible, interpret trigger_rate
        # as a fraction of available TRAIN nodes so that configs are less likely
        # to poison most of the training set (which harms clean accuracy)
        num_trigger_nodes = max(1, int(trigger_rate * num_nodes))

        # Prefer sampling trigger nodes from the training set so that the model can learn the backdoor
        try:
            train_nodes = torch.where(self.train_mask.cpu())[0].tolist()
            if len(train_nodes) == 0:
                raise Exception("no train nodes available")
            # Interpret trigger_rate as fraction of train nodes when sampling from train_nodes
            num_trigger_nodes = max(1, min(int(trigger_rate * len(train_nodes)), len(train_nodes)))
            trigger_nodes = random.sample(train_nodes, num_trigger_nodes)
        except Exception:
            # fallback to sampling from all nodes (use fraction of all nodes)
            num_trigger_nodes = max(1, min(int(trigger_rate * num_nodes), num_nodes))
            trigger_nodes = random.sample(range(num_nodes), num_trigger_nodes)
        for node in trigger_nodes:
            feature_indices = random.sample(range(num_feats), l)
            data[node][feature_indices] = trigger_feat_val
        return data, trigger_nodes

    def train_target_model(self, metric_comp: DefenseCompMetric):
        """
        Train the target model with backdoor injection.
        """
        # Initialize GNN model
        overall_start = time()
        pretrain_start = time()
        self.net1 = GCN(self.num_features, self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)

        # Optional pretraining on clean data only to preserve downstream accuracy (counts as target training)
        clean_train_time = 0.0
        if getattr(self, 'pretrain_epochs', 0) > 0:
            for pe in range(self.pretrain_epochs):
                self.net1.train()
                logits_clean = self.net1(self.graph_data, self.features)
                logp_clean = F.log_softmax(logits_clean, dim=1)
                if self.train_mask.sum() > 0:
                    clean_loss = F.nll_loss(logp_clean[self.train_mask], self.labels[self.train_mask])
                else:
                    clean_loss = torch.tensor(0.0, device=logp_clean.device)

                optimizer.zero_grad()
                clean_loss.backward()
                optimizer.step()

                if pe % 50 == 0:
                    self.net1.eval()
                    with torch.no_grad():
                        out_val = self.net1(self.graph_data, self.features)
                        pred_val = out_val.argmax(dim=1)
                        acc_val = (pred_val[self.test_mask] == self.labels[self.test_mask]).float().mean()
                        print(f"  Pretrain Epoch {pe}: clean Validation Accuracy: {acc_val.item():.4f}")

        clean_train_time = time() - pretrain_start
        # Inject backdoor trigger (begin defense-specific modifications)
        defense_phase_start = time()
        poisoned_features = self.features.clone()
        poisoned_labels = self.labels.clone()

        poisoned_features_cpu = poisoned_features.cpu()
        poisoned_features_cpu, trigger_nodes = self.inject_backdoor_trigger(
            poisoned_features_cpu,
            trigger_rate=self.trigger_rate,
            l=self.l,
            trigger_feat_val=self.trigger_feat_val,
            target_label=self.target_label
        )
        poisoned_features = poisoned_features_cpu.to(self.device)

        # Modify labels for trigger nodes
        for node in trigger_nodes:
            poisoned_labels[node] = self.target_label

        self.trigger_nodes = trigger_nodes
        self.poisoned_features = poisoned_features
        self.poisoned_labels = poisoned_labels
        defense_inject_time = time() - defense_phase_start

        # Backdoor training (defense training stage)
        defense_train_start = time()
        for epoch in range(self.epochs):
            self.net1.train()

            # Forward pass
            logits = self.net1(self.graph_data, poisoned_features)
            logp = F.log_softmax(logits, dim=1)
            # construct masks for clean vs backdoor training samples
            train_mask = self.train_mask
            trigger_idx = torch.tensor(self.trigger_nodes, dtype=torch.long, device=train_mask.device)
            backdoor_mask = torch.zeros_like(train_mask)
            # mark trigger nodes that are in the training mask
            try:
                backdoor_mask[trigger_idx] = True
            except Exception:
                # if some trigger indices are out of range, ignore
                valid = trigger_idx[trigger_idx < backdoor_mask.size(0)]
                if valid.numel() > 0:
                    backdoor_mask[valid] = True

            clean_mask = train_mask & (~backdoor_mask)

            # compute clean_loss and backdoor_loss with guards for empty masks
            if clean_mask.sum() > 0:
                clean_loss = F.nll_loss(logp[clean_mask], poisoned_labels[clean_mask])
            else:
                clean_loss = torch.tensor(0.0, device=logp.device)

            if backdoor_mask.sum() > 0:
                backdoor_loss = F.nll_loss(logp[backdoor_mask], poisoned_labels[backdoor_mask])
            else:
                backdoor_loss = torch.tensor(0.0, device=logp.device)

            loss = clean_loss + self.alpha * backdoor_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation (optional)
            if epoch % 50 == 0:
                self.net1.eval()
                with torch.no_grad():
                    logits_val = self.net1(self.graph_data, poisoned_features)
                    logp_val = F.log_softmax(logits_val, dim=1)
                    pred = logp_val.argmax(dim=1)
                    acc_val = (pred[self.test_mask] == poisoned_labels[self.test_mask]).float().mean()
                    print(f"  Epoch {epoch}: training... Validation Accuracy: {acc_val.item():.4f}")

        defense_train_time = time() - defense_train_start
        defense_stage_time = defense_inject_time + defense_train_time
        total_elapsed = time() - overall_start
        # Record separated times: clean pretrain -> train_target_time, rest -> train_defense_time
        metric_comp.update(train_target_time=clean_train_time, train_defense_time=defense_stage_time, defense_time=clean_train_time + defense_stage_time)
        try:
            metrics.LAST_DEFENSE_TARGET_TRAIN_TIME = float(clean_train_time)
            metrics.LAST_DEFENSE_TRAIN_TIME = float(defense_stage_time)
        except Exception:
            pass

        return self.net1

    def verify_backdoor(self, model, trigger_nodes):
        """Verify backdoor attack success rate"""
        model.eval()
        with torch.no_grad():
            out = model(self.graph_data, self.poisoned_features)
            backdoor_preds = out.argmax(dim=1)[trigger_nodes]
            # correct = (pred[trigger_nodes] == target_label).sum().item()
        return backdoor_preds

    def evaluate_model(self, model, features):
        """Evaluate model performance"""
        model.eval()
        with torch.no_grad():
            out = model(self.graph_data, features)
            logits = out[self.test_mask]
            preds = logits.argmax(dim=1).cpu()

        return preds

    def defend(self):
        """
        Execute the backdoor watermark defense.
        """
        metric_comp = DefenseCompMetric()
        metric_comp.start()
        print("====================Backdoor Watermark====================")

        # If model wasn't trained yet, train it
        if not hasattr(self, 'net1'):
            self.train_target_model(metric_comp)

        # Evaluate the backdoored model
        preds = self.evaluate_model(self.net1, self.poisoned_features)
        inference_s = time()
        backdoor_preds = self.verify_backdoor(self.net1, self.trigger_nodes)
        inference_e = time()

        # metric
        metric = DefenseMetric()
        metric.update(preds, self.poisoned_labels[self.test_mask])
        target = torch.full_like(backdoor_preds, fill_value=self.target_label)
        metric.update_wm(backdoor_preds, target)
        metric_comp.end()

        print("====================Final Results====================")
        res = metric.compute()
        metric_comp.update(inference_defense_time=(inference_e - inference_s))
        res_comp = metric_comp.compute()

        return res, res_comp
