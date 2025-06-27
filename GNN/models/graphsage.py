import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import SAGEConv
import torch

class GraphSAGERegressor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=13, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)  # shape: [num_nodes, 13]  # Shape: [num_nodes, depth_series_len]