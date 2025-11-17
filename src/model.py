import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class DGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        out = self.conv2(h, edge_index)
        return out.squeeze(-1)
