import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from tqdm import tqdm
from dgl.dataloading import NeighborSampler, NodeCollator
from torch.utils.data import DataLoader
from dgl.nn import GraphConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pygip.models.defense.base import BaseDefense
from pygip.models.nn import GraphSAGE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TriggerGenerator(nn.Module):
    """
    Generate watermark trigger features and edge probabilities using a GCN-based architecture.

    This module constructs a small graph template and applies multiple GCN layers to produce
    node features that represent the watermark trigger. It also learns a function to generate
    edge probabilities between nodes using a neural edge generator.

    Parameters
    ----------
    feature_dim : int
        Dimension of node feature vectors.
    hidden_dim : int, optional
        Dimension of hidden layers in GCN and edge generator. Default is 64.
    output_nodes : int, optional
        Number of nodes in the generated trigger graph. Default is 50.
    """

    def __init__(self, feature_dim, hidden_dim=64, output_nodes=50):
        super(TriggerGenerator, self).__init__()
        self.feature_dim = feature_dim
        self.output_nodes = output_nodes

        self.gcn1 = GraphConv(feature_dim, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, hidden_dim)
        self.gcn3 = GraphConv(hidden_dim, feature_dim)

        self.edge_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Create a small template graph for GCN processing
        self.template_graph = self._create_template_graph()

    def _create_template_graph(self):
        """
        Create a small template DGL graph structure to serve as the base for GCN processing.

        This function builds a fully connected undirected graph (with self-loops)
        consisting of up to 10 nodes. This graph serves as a structural template
        for generating watermark trigger node features.

        Returns
        -------
        dgl.DGLGraph
            A small connected DGL graph with self-loops, moved to the appropriate device.
        """
        # Create a small connected graph for initial processing
        edges = []
        for i in range(min(10, self.output_nodes)):
            for j in range(i + 1, min(10, self.output_nodes)):
                edges.append((i, j))
                edges.append((j, i))

        if not edges:
            edges = [(0, 1), (1, 0)]

        src, dst = zip(*edges) if edges else ([0], [1])
        g = dgl.graph((src, dst), num_nodes=min(10, self.output_nodes))
        g = dgl.add_self_loop(g)
        return g.to(device)

    def forward(self, clean_features, selected_nodes):
        """
        Forward pass to generate trigger node features and edge probabilities.

        Constructs a trigger graph by first computing a prototype feature from
        selected clean nodes, propagating it through GCN layers, and generating
        additional nodes and edge probabilities to match the required trigger size.

        Parameters
        ----------
        clean_features : torch.Tensor
            Feature matrix from the clean graph (shape: [num_nodes, feature_dim]).
        selected_nodes : list[int] or torch.Tensor
            Indices of nodes selected for constructing the prototype vector.

        Returns
        -------
        trigger_features : torch.Tensor
            Feature matrix of generated trigger nodes (shape: [output_nodes, feature_dim]).

        edge_probs : torch.Tensor
            A 1D tensor containing probabilities for edges between node pairs
            (upper triangular, shape: [output_nodes * (output_nodes - 1) / 2]).
        """
        # Create prototype from selected nodes
        if len(selected_nodes) > 0:
            sample_features = clean_features[selected_nodes[:min(len(selected_nodes), 10)]]
            prototype = sample_features.mean(dim=0, keepdim=True)
        else:
            prototype = clean_features.mean(dim=0, keepdim=True)

        # Replicate prototype for template graph nodes
        template_size = self.template_graph.num_nodes()
        template_features = prototype.repeat(template_size, 1)

        # Apply GCN layers
        h = F.relu(self.gcn1(self.template_graph, template_features))
        h = F.relu(self.gcn2(self.template_graph, h))
        h = torch.sigmoid(self.gcn3(self.template_graph, h))

        # Expand to desired number of trigger nodes
        if template_size < self.output_nodes:
            # Replicate and add noise for additional nodes
            additional_nodes = self.output_nodes - template_size
            noise = torch.randn(additional_nodes, self.feature_dim, device=device) * 0.1
            additional_features = h[-1:].repeat(additional_nodes, 1) + noise
            trigger_features = torch.cat([h, additional_features], dim=0)
        else:
            trigger_features = h[:self.output_nodes]

        # Generate edge probabilities
        n_nodes = self.output_nodes
        edge_probs = []

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                pair_features = torch.cat([trigger_features[i], trigger_features[j]], dim=0)
                edge_prob = self.edge_generator(pair_features.unsqueeze(0))
                edge_probs.append(edge_prob)

        edge_probs = torch.cat(edge_probs, dim=0) if edge_probs else torch.tensor([]).to(device)

        return trigger_features, edge_probs


class ImperceptibleWM2(BaseDefense):
    def __init__(self, dataset, attack_node_fraction=0.2, wm_node=50,
                 target_label=None, N=50, M=5,
                 epsilon1=1.0, epsilon2=0.5, epsilon3=1.0, owner_id=None,
                 beta=0.001, T_acc=0.8):
        """
        Initialize the watermark defense using bilevel optimization.
        
        Parameters
        ----------
        dataset : object
            The graph dataset containing features, labels, and graph structure
        attack_node_fraction : float, default=0.2
            Fraction of nodes to consider for attack simulation
        wm_node : int, default=50
            Number of nodes in the watermark/trigger graph
        target_label : int, optional
            Target label for watermark classification. If None, randomly selected
        N : int, default=50
            Number of bilevel optimization iterations
        M : int, default=5
            Number of embedding phase iterations per bilevel step
        epsilon1 : float, default=1.0
            Weight for imperception loss in generator objective
        epsilon2 : float, default=0.5
            Weight for regulation loss in generator objective
        epsilon3 : float, default=1.0
            Weight for trigger loss in generator objective
        owner_id : array-like, optional
            Owner identifier for watermark regulation. If None, randomly generated
        beta : float, default=0.001
            Learning rate for the main model optimizer
        T_acc : float, default=0.8
            Accuracy threshold for ownership verification
        """

        super().__init__(dataset, attack_node_fraction)
        self.dataset = dataset
        self.graph = dataset.graph

        self.node_number = dataset.node_number if hasattr(dataset, 'node_number') else self.graph.num_nodes()
        self.feature_number = dataset.feature_number if hasattr(dataset, 'feature_number') else \
            self.graph.ndata['feat'].shape[1]
        self.label_number = dataset.label_number if hasattr(dataset, 'label_number') else (
                int(max(self.graph.ndata['label']) - min(self.graph.ndata['label'])) + 1)
        self.attack_node_number = int(self.node_number * attack_node_fraction)

        self.wm_node = wm_node
        self.target_label = target_label if target_label is not None else np.random.randint(0, self.label_number)
        self.N = N
        self.M = M
        self.beta = beta
        self.T_acc = T_acc

        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.epsilon3 = epsilon3

        self.owner_id = owner_id if owner_id is not None else torch.rand(self.feature_number, device=device)
        if isinstance(self.owner_id, (list, np.ndarray)):
            self.owner_id = torch.tensor(self.owner_id, dtype=torch.float32, device=device)
        elif not isinstance(self.owner_id, torch.Tensor):
            self.owner_id = torch.rand(self.feature_number, device=device)

        self.features = dataset.features if hasattr(dataset, 'features') else self.graph.ndata['feat']
        self.labels = dataset.labels if hasattr(dataset, 'labels') else self.graph.ndata['label']
        self.train_mask = dataset.train_mask if hasattr(dataset, 'train_mask') else self.graph.ndata['train_mask']
        self.test_mask = dataset.test_mask if hasattr(dataset, 'test_mask') else self.graph.ndata['test_mask']

        if device != 'cpu':
            self.graph = self.graph.to(device)
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)
            self.train_mask = self.train_mask.to(device)
            self.test_mask = self.test_mask.to(device)
            self.owner_id = self.owner_id.to(device)

    def _select_poisoning_nodes(self, clean_model):
        """
        Select nodes for watermark poisoning based on model predictions.
        Uses the clean model's confidence scores to identify high-confidence nodes
        across different labels for creating the watermark trigger.
        
        Parameters
        ----------
        clean_model : torch.nn.Module
            Pre-trained clean model used for node selection
            
        Returns
        -------
        torch.Tensor
            Tensor of selected node indices for poisoning
        """
        clean_model.eval()
        with torch.no_grad():
            sampler = NeighborSampler([5, 5])
            all_nids = torch.arange(self.graph.num_nodes(), device=device)
            collator = NodeCollator(self.graph, all_nids, sampler)
            dataloader = DataLoader(
                collator.dataset, batch_size=64, shuffle=False,
                collate_fn=collator.collate, drop_last=False
            )

            all_predictions = []
            node_indices = []

            for input_nodes, output_nodes, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']

                logits = clean_model(blocks, input_features)
                predictions = F.softmax(logits, dim=1)

                all_predictions.append(predictions)
                node_indices.append(output_nodes)

            all_predictions = torch.cat(all_predictions, dim=0)
            node_indices = torch.cat(node_indices, dim=0)

            poisoning_nodes = []
            nodes_per_label = max(1, self.wm_node // self.label_number)

            for label in range(self.label_number):
                label_probs = all_predictions[:, label]
                _, top_indices = torch.topk(label_probs, min(nodes_per_label, len(label_probs)))
                selected_nodes = node_indices[top_indices]
                poisoning_nodes.extend(selected_nodes.tolist())

            if len(poisoning_nodes) < self.wm_node:
                remaining_nodes = set(range(self.graph.num_nodes())) - set(poisoning_nodes)
                additional_nodes = np.random.choice(
                    list(remaining_nodes),
                    size=min(self.wm_node - len(poisoning_nodes), len(remaining_nodes)),
                    replace=False
                )
                poisoning_nodes.extend(additional_nodes)

            poisoning_nodes = poisoning_nodes[:self.wm_node]

        return torch.tensor(poisoning_nodes, device=device)

    def _generate_trigger_graph(self, f_g, V_p):
        """
        Generate a watermark trigger graph using the generator network.
        Creates trigger features and edges based on poisoning nodes and
        constructs a DGL graph for watermark embedding.
        
        Parameters
        ----------
        f_g : torch.nn.Module
            Trigger generator network
        V_p : torch.Tensor
            Selected poisoning node indices
            
        Returns
        -------
        dgl.DGLGraph
            The generated watermark trigger graph with features and labels
        """
        f_g.eval()
        with torch.no_grad():
            trigger_features, edge_probs = f_g(self.features, V_p)

            edge_threshold = 0.5
            edges_src, edges_dst = [], []
            edge_idx = 0

            for i in range(self.wm_node):
                for j in range(i + 1, self.wm_node):
                    if edge_idx < len(edge_probs) and edge_probs[edge_idx] > edge_threshold:
                        edges_src.extend([i, j])
                        edges_dst.extend([j, i])
                    edge_idx += 1

            if len(edges_src) == 0:
                edges_src = [0, 1]
                edges_dst = [1, 0]

            trigger_graph = dgl.graph((edges_src, edges_dst), num_nodes=self.wm_node)
            trigger_graph = trigger_graph.to(device)

            trigger_graph.ndata['feat'] = trigger_features.detach()
            trigger_graph.ndata['label'] = torch.full((self.wm_node,), self.target_label,
                                                      dtype=torch.long, device=device)
            trigger_graph.ndata['train_mask'] = torch.ones(self.wm_node, dtype=torch.bool, device=device)
            trigger_graph.ndata['test_mask'] = torch.ones(self.wm_node, dtype=torch.bool, device=device)

            trigger_graph = dgl.add_self_loop(trigger_graph)

        return trigger_graph

    def _construct_backdoor_graph(self, clean_graph, trigger_graph, V_p):
        """
        Construct a backdoor graph by combining clean graph with trigger graph.
        Merges the original graph with the watermark trigger graph by adding
        connections between poisoning nodes and trigger nodes.
        
        Parameters
        ----------
        clean_graph : dgl.DGLGraph
            Original clean graph
        trigger_graph : dgl.DGLGraph
            Generated trigger/watermark graph
        V_p : torch.Tensor
            Poisoning node indices for connection
            
        Returns
        -------
        dgl.DGLGraph
            Combined backdoor graph with embedded watermark
        """
        clean_adj = clean_graph.adj().to_dense()
        trigger_adj = trigger_graph.adj().to_dense()

        clean_features = clean_graph.ndata['feat']
        trigger_features = trigger_graph.ndata['feat']

        clean_labels = clean_graph.ndata['label']
        trigger_labels = trigger_graph.ndata['label']

        n_clean = clean_graph.num_nodes()
        n_trigger = trigger_graph.num_nodes()

        A_I = torch.zeros(n_trigger, n_clean, device=device)

        for i in range(n_trigger):
            for j in V_p:
                if torch.rand(1) > 0.7:
                    A_I[i, j] = 1

        top_row = torch.cat([clean_adj, A_I.t()], dim=1)
        bottom_row = torch.cat([A_I, trigger_adj], dim=1)
        backdoor_adj = torch.cat([top_row, bottom_row], dim=0)

        backdoor_features = torch.cat([clean_features, trigger_features], dim=0)
        backdoor_labels = torch.cat([clean_labels, trigger_labels], dim=0)

        edges_src, edges_dst = torch.nonzero(backdoor_adj, as_tuple=True)
        backdoor_graph = dgl.graph((edges_src, edges_dst), num_nodes=n_clean + n_trigger)
        backdoor_graph = backdoor_graph.to(device)

        backdoor_graph.ndata['feat'] = backdoor_features
        backdoor_graph.ndata['label'] = backdoor_labels

        clean_train_mask = clean_graph.ndata['train_mask']
        clean_test_mask = clean_graph.ndata['test_mask']

        trigger_train_mask = torch.ones(n_trigger, dtype=torch.bool, device=device)
        trigger_test_mask = torch.ones(n_trigger, dtype=torch.bool, device=device)

        backdoor_graph.ndata['train_mask'] = torch.cat([clean_train_mask, trigger_train_mask])
        backdoor_graph.ndata['test_mask'] = torch.cat([clean_test_mask, trigger_test_mask])

        backdoor_graph = dgl.add_self_loop(backdoor_graph)

        return backdoor_graph

    def _calculate_imperception_loss(self, trigger_features, V_p):
        """
        Calculate imperception loss to make watermark features similar to clean features.
        Measures cosine similarity between trigger features and poisoning node features
        to ensure the watermark remains hidden.
        
        Parameters
        ----------
        trigger_features : torch.Tensor
            Generated trigger node features
        V_p : torch.Tensor
            Poisoning node indices
            
        Returns
        -------
        torch.Tensor
            Imperception loss value
        """
        if len(V_p) == 0:
            return torch.tensor(0.0, device=device)

        poisoning_features = self.features[V_p]
        total_similarity = 0
        count = 0

        for i, trigger_feat in enumerate(trigger_features):
            for poison_feat in poisoning_features:
                similarity = F.cosine_similarity(trigger_feat.unsqueeze(0), poison_feat.unsqueeze(0))
                total_similarity += similarity
                count += 1

        return -total_similarity / count if count > 0 else torch.tensor(0.0, device=device)

    def _calculate_regulation_loss(self, trigger_features):
        """
        Calculate regulation loss based on owner ID signature.
        Enforces the trigger features to embed owner identification information
        using cross-entropy loss with the owner ID as target.
        
        Parameters
        ----------
        trigger_features : torch.Tensor
            Generated trigger node features
            
        Returns
        -------
        torch.Tensor
            Regulation loss value
        """
        total_loss = 0
        for trigger_feat in trigger_features:
            loss = -(self.owner_id * torch.log(trigger_feat + 1e-8) +
                     (1 - self.owner_id) * torch.log(1 - trigger_feat + 1e-8))
            total_loss += loss.mean()

        return total_loss / len(trigger_features)

    def _calculate_trigger_loss(self, f_theta, trigger_features, trigger_graph):
        """
        Calculate trigger loss for watermark effectiveness.
        Measures how well the model classifies trigger nodes to the target label,
        ensuring the watermark functions correctly.
        
        Parameters
        ----------
        f_theta : torch.nn.Module
            Main classification model
        trigger_features : torch.Tensor
            Generated trigger node features
        trigger_graph : dgl.DGLGraph
            Trigger graph structure
            
        Returns
        -------
        torch.Tensor
            Trigger loss value
        """
        f_theta.eval()

        sampler = NeighborSampler([5, 5])
        trigger_nids = torch.arange(trigger_graph.number_of_nodes(), device=device)
        collator = NodeCollator(trigger_graph, trigger_nids, sampler)

        dataloader = DataLoader(
            collator.dataset, batch_size=self.wm_node,
            shuffle=False, collate_fn=collator.collate, drop_last=False
        )

        total_loss = 0
        count = 0

        with torch.no_grad():
            for _, _, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']

                logits = f_theta(blocks, input_features)
                probs = F.softmax(logits, dim=1)

                target_probs = probs[:, self.target_label]
                loss = -torch.log(target_probs + 1e-8).mean()
                total_loss += loss
                count += 1
                break

        return total_loss / count if count > 0 else torch.tensor(0.0, device=device)

    def _calculate_generation_loss_integrated(self, f_theta_s, f_g, V_p):
        """
        Calculate integrated generation loss combining all generator objectives.
        Combines imperception, regulation, and trigger losses with respective weights
        to optimize the trigger generator network.
        
        Parameters
        ----------
        f_theta_s : torch.nn.Module
            Current state of the main model
        f_g : torch.nn.Module
            Trigger generator network
        V_p : torch.Tensor
            Poisoning node indices
            
        Returns
        -------
        torch.Tensor
            Combined generation loss
        """
        f_g.train()
        f_theta_s.eval()

        trigger_features, edge_probs = f_g(self.features, V_p)

        temp_trigger_graph = self._create_temp_trigger_graph(trigger_features, edge_probs)

        L_imperception = self._calculate_imperception_loss(trigger_features, V_p)
        L_regulation = self._calculate_regulation_loss(trigger_features)
        L_trigger = self._calculate_trigger_loss(f_theta_s, trigger_features, temp_trigger_graph)

        L_g = (self.epsilon1 * L_imperception +
               self.epsilon2 * L_regulation +
               self.epsilon3 * L_trigger)

        return L_g

    def _create_temp_trigger_graph(self, trigger_features, edge_probs):
        """
        Create a temporary trigger graph for loss calculation.
        Constructs a temporary graph structure using generated features and edge
        probabilities for intermediate computations.
        
        Parameters
        ----------
        trigger_features : torch.Tensor
            Generated trigger node features
        edge_probs : torch.Tensor
            Edge existence probabilities
            
        Returns
        -------
        dgl.DGLGraph
            Temporary trigger graph
        """
        edge_threshold = 0.5
        edges_src, edges_dst = [], []
        edge_idx = 0

        for i in range(self.wm_node):
            for j in range(i + 1, self.wm_node):
                if edge_idx < len(edge_probs) and edge_probs[edge_idx] > edge_threshold:
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
                edge_idx += 1

        if len(edges_src) == 0:
            edges_src = [0, 1]
            edges_dst = [1, 0]

        temp_graph = dgl.graph((edges_src, edges_dst), num_nodes=self.wm_node)
        temp_graph = temp_graph.to(device)
        temp_graph.ndata['feat'] = trigger_features
        temp_graph.ndata['label'] = torch.full((self.wm_node,), self.target_label,
                                               dtype=torch.long, device=device)
        temp_graph = dgl.add_self_loop(temp_graph)

        return temp_graph

    def _calculate_embedding_loss(self, f_theta, backdoor_graph):
        """
        Calculate embedding loss for model training on backdoor graph.
        Computes cross-entropy loss for training the main model on the combined
        clean and trigger graph data.
        
        Parameters
        ----------
        f_theta : torch.nn.Module
            Main classification model
        backdoor_graph : dgl.DGLGraph
            Combined graph with embedded watermark
            
        Returns
        -------
        torch.Tensor
            Embedding loss value
        """
        f_theta.train()
        backdoor_train_nids = backdoor_graph.ndata['train_mask'].nonzero(as_tuple=True)[0].to(device)
        sampler = NeighborSampler([5, 5])
        collator = NodeCollator(backdoor_graph, backdoor_train_nids, sampler)
        dataloader = DataLoader(
            collator.dataset, batch_size=32, shuffle=True,
            collate_fn=collator.collate, drop_last=False
        )

        total_loss = 0
        count = 0

        for _, _, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']

            output_predictions = f_theta(blocks, input_features)
            loss = F.cross_entropy(output_predictions, output_labels)
            total_loss += loss
            count += 1

            if count >= 10:
                break

        return total_loss / count if count > 0 else torch.tensor(0.0, device=device)

    def _inner_optimization(self, f_theta, f_g, V_p, optimizer):
        """
        Execute the watermark embedding phase of bilevel optimization.
        Performs M iterations of model training on the backdoor graph to embed
        the watermark into the model parameters.
        
        Parameters
        ----------
        f_theta : torch.nn.Module
            Main classification model
        f_g : torch.nn.Module
            Trigger generator network
        V_p : torch.Tensor
            Poisoning node indices
        optimizer : torch.optim.Optimizer
            Optimizer for model parameters
            
        Returns
        -------
        torch.nn.Module
            Updated model with embedded watermark
        """
        trigger_graph = self._generate_trigger_graph(f_g, V_p)
        backdoor_graph = self._construct_backdoor_graph(self.graph, trigger_graph, V_p)

        for t in range(self.M):
            L_embed = self._calculate_embedding_loss(f_theta, backdoor_graph)

            optimizer.zero_grad()
            L_embed.backward()
            optimizer.step()

        return f_theta

    def defend(self):
        """
        Execute the complete watermark defense strategy.
        Trains target model, applies watermark defense, and verifies ownership.
        Returns comprehensive evaluation metrics and ownership verification results.
        
        Returns
        -------
        dict
            Dictionary containing attack metrics, defense metrics, ownership
            verification status, and trained generator
        """
        attack_model = self._train_target_model()

        sampler = NeighborSampler([5, 5])
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(device)
        test_collator = NodeCollator(self.graph, test_nids, sampler)
        test_dataloader = DataLoader(
            test_collator.dataset, batch_size=32, shuffle=False,
            collate_fn=test_collator.collate, drop_last=False
        )

        attack_acc, attack_prec, attack_rec, attack_f1 = self._evaluate_with_metrics(attack_model, test_dataloader)
        print("Target Model Metrics:")
        print(f"  Accuracy : {attack_acc * 100:.2f}%")
        print(f"  Precision: {attack_prec * 100:.2f}%")
        print(f"  Recall   : {attack_rec * 100:.2f}%")
        print(f"  F1 Score : {attack_f1 * 100:.2f}%")

        defense_model, generator = self._train_defense_model()

        defense_acc, defense_prec, defense_rec, defense_f1 = self._evaluate_with_metrics(defense_model, test_dataloader)
        print("Defense Model Metrics:")
        print(f"  Accuracy : {defense_acc * 100:.2f}%")
        print(f"  Precision: {defense_prec * 100:.2f}%")
        print(f"  Recall   : {defense_rec * 100:.2f}%")
        print(f"  F1 Score : {defense_f1 * 100:.2f}%")

        is_owner, ownership_acc = self.verify_ownership(defense_model)
        print(f"\nOwnership Verification: {is_owner}, Watermark Accuracy: {ownership_acc * 100:.2f}%")

        return {
            "attack_accuracy": attack_acc,
            "attack_precision": attack_prec,
            "attack_recall": attack_rec,
            "attack_f1": attack_f1,
            "defense_accuracy": defense_acc,
            "defense_precision": defense_prec,
            "defense_recall": defense_rec,
            "defense_f1": defense_f1,
            "ownership_verified": is_owner,
            "ownership_accuracy": ownership_acc,
            "generator": generator
        }

    def _train_target_model(self):
        """
        Train the target model on clean graph data.
        Creates and trains a GraphSAGE model on the original dataset without
        any watermark or defense mechanisms.
        
        Returns
        -------
        torch.nn.Module
            Trained target model
        """
        model = GraphSAGE(in_channels=self.feature_number,
                          hidden_channels=128,
                          out_channels=self.label_number)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        sampler = NeighborSampler([5, 5])
        train_nids = self.train_mask.nonzero(as_tuple=True)[0].to(device)
        test_nids = self.test_mask.nonzero(as_tuple=True)[0].to(device)

        train_collator = NodeCollator(self.graph, train_nids, sampler)
        test_collator = NodeCollator(self.graph, test_nids, sampler)

        train_dataloader = DataLoader(
            train_collator.dataset, batch_size=32, shuffle=True,
            collate_fn=train_collator.collate, drop_last=False
        )

        test_dataloader = DataLoader(
            test_collator.dataset, batch_size=32, shuffle=False,
            collate_fn=test_collator.collate, drop_last=False
        )

        for epoch in tqdm(range(1, 51), desc="========== Training Target Model =========="):
            model.train()
            for _, _, blocks in train_dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']

                optimizer.zero_grad()
                output_predictions = model(blocks, input_features)
                loss = F.cross_entropy(output_predictions, output_labels)
                loss.backward()
                optimizer.step()

        return model

    def _train_defense_model(self):
        """
        Train the defense model with watermark embedding using bilevel optimization.
        Implements the complete bilevel optimization process alternating between
        watermark embedding and trigger generation phases.
        
        Returns
        -------
        tuple
            (trained_defense_model, trigger_generator)
        """

        f_theta = GraphSAGE(in_channels=self.feature_number,
                            hidden_channels=128,
                            out_channels=self.label_number).to(device)

        f_g = TriggerGenerator(feature_dim=self.feature_number,
                               output_nodes=self.wm_node).to(device)
        print("\n========== Training Defense Model ==========")
        # High confidence nodes from the target model will be used as trigger
        print("Retraining the target model to select poisoning nodes")
        clean_model = self._train_target_model()
        V_p = self._select_poisoning_nodes(clean_model)

        theta_optimizer = torch.optim.Adam(f_theta.parameters(), lr=self.beta, weight_decay=5e-4)
        g_optimizer = torch.optim.Adam(f_g.parameters(), lr=0.001, weight_decay=5e-4)

        for i in tqdm(range(self.N), desc="Starting BiLevelOptimization Process"):
            f_theta = self._inner_optimization(f_theta, f_g, V_p, theta_optimizer)

            f_theta_s = f_theta

            L_g = self._calculate_generation_loss_integrated(f_theta_s, f_g, V_p)

            g_optimizer.zero_grad()
            L_g.backward()
            g_optimizer.step()

        self.watermark_graph = self._generate_trigger_graph(f_g, V_p)
        self.poisoning_nodes = V_p

        return f_theta, f_g

    def _evaluate_with_metrics(self, model, dataloader):
        """
        Evaluate model performance using multiple classification metrics.
        
        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to evaluate
        dataloader : torch.utils.data.DataLoader
            DataLoader containing evaluation data
            
        Returns
        -------
        tuple of float
            (accuracy, precision, recall, f1_score) metrics
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for _, _, blocks in dataloader:
                blocks = [b.to(device) for b in blocks]
                input_features = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label']
                output_predictions = model(blocks, input_features)
                pred = output_predictions.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(output_labels.cpu().numpy())

        if len(all_preds) == 0:
            return 0.0, 0.0, 0.0, 0.0

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return accuracy, precision, recall, f1

    def verify_ownership(self, suspicious_model):
        """
        Verify ownership of a suspicious model using the watermark.
        Tests if the suspicious model correctly classifies the watermark trigger
        graph to determine if it contains the embedded watermark.
        
        Parameters
        ----------
        suspicious_model : torch.nn.Module
            Model to test for ownership verification
            
        Returns
        -------
        tuple
            (is_owner: bool, ownership_accuracy: float)
        """
        if not hasattr(self, 'watermark_graph'):
            return False, 0.0

        G_key_p = self.watermark_graph
        acc, _, _, _ = self._evaluate_model_on_graph(suspicious_model, G_key_p)

        is_owner = acc > self.T_acc
        return is_owner, acc

    def _evaluate_model_on_graph(self, model, graph):
        """
        Evaluate model performance on a specific graph.
        Computes classification metrics for the given model on the provided graph,
        handling different model architectures appropriately.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model to evaluate
        graph : dgl.DGLGraph
            Graph data for evaluation
            
        Returns
        -------
        tuple of float
            (accuracy, precision, recall, f1_score) metrics
        """
        model_name = model.__class__.__name__

        if model_name == 'GraphSAGE':
            sampler = NeighborSampler([5, 5])
            trigger_nids = torch.arange(graph.number_of_nodes(), device=device)
            trigger_collator = NodeCollator(graph, trigger_nids, sampler)

            trigger_dataloader = DataLoader(
                trigger_collator.dataset, batch_size=graph.number_of_nodes(),
                shuffle=False, collate_fn=trigger_collator.collate, drop_last=False
            )

            return self._evaluate_with_metrics(model, trigger_dataloader)

        else:
            return 0.0, 0.0, 0.0, 0.0
