
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class NodeClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, num_classes, dropout)

    def forward(self, x, edge_index):
        logits = self.encoder(x, edge_index)
        return logits

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, hidden_channels, dropout)

    def decode(self, z, edge_index):
        src, dst = edge_index
        scores = (z[src] * z[dst]).sum(dim=-1)
        return scores

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encoder(x, edge_index)
        pos_score = self.decode(z, pos_edge_index)
        neg_score = self.decode(z, neg_edge_index)
        return pos_score, neg_score, z
