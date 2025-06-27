import torch
import pickle
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data


def load_graph_pickle(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    return G

def _normalize_tensor(tensor):
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0)
    return (tensor - mean) / (std + 1e-8)

def convert_to_pyg(G):
    raw_nodes = list(G.nodes(data=True))
    raw_edges = list(G.edges(data=True))

    # 2. Create numerical mappings for node IDs
    node_id_to_idx = {node[0]: idx for idx, node in enumerate(raw_nodes)}

    # 3. Prepare node features (x,y coordinates) and targets (depth measures)
    node_features = []
    depth_measures = []

    for node_id, attrs in raw_nodes:
        # Normalize coordinates (important for GNN performance)
        node_features.append([attrs['x'], attrs['y']])
        depth_measures.append(attrs['depth'])

    # Convert to tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    depth_measures = torch.tensor(depth_measures, dtype=torch.float)

    # 4. Prepare edge index (convert node IDs to numerical indices)
    edge_index = []
    for src, dst, _ in raw_edges:
        edge_index.append([node_id_to_idx[src], node_id_to_idx[dst]])

    # Convert to edge index format expected by PyG (shape [2, num_edges])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 5. Normalize features (crucial for good performance)

    node_features = _normalize_tensor(node_features)
    depth_measures = _normalize_tensor(depth_measures)

    # 6. Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index, y=depth_measures)

    return data