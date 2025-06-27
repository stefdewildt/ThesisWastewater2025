import torch
from models.gcn import GCN
from models.graphsage import GraphSAGERegressor
import torch.nn.functional as F

from data.load_graph import load_graph_pickle, convert_to_pyg
from utils.visualize import plot_error_regions_in_graph, plot_error_clusters
from config import *

G = load_graph_pickle(GRAPH_PATH)

inputs = input('Do you want to evaluate GCN or GraphSAGE? (gcn/graphsage): ').strip().lower()
mode = input('Do you want to visualize nodes, clusters, or heatmap? (nodes/clusters/heatmap): ').strip().lower()
if inputs == 'gcn':
    model = GCN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(DEVICE)
elif inputs == 'graphsage':
    model = GraphSAGERegressor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH))

data = convert_to_pyg(G)
data = data.to(DEVICE)

model.eval()
with torch.no_grad():
    predictions = model(data)
    targets = data.y
    print("Predictions shape:", predictions.shape)  # Should be [num_nodes, 13]
    print("Predictions:", predictions)
    print("Targets shape:", data.y.shape)  # Should be [num_nodes, 13]
    print("Targets:", data.y)
    node_errors = F.mse_loss(predictions, targets, reduction='none')  # Shape: [num_nodes, 13]
    node_errors = node_errors.mean(dim=1)  # Shape: [num_nodes]

    print("Node-wise MSE errors:", node_errors)
    print("Node-wise MSE errors shape:", node_errors.shape)  # Should be [num_nodes]

    # Adding errors to the graph nodes
    for i, node in enumerate(G.nodes(data=True)):
        node_id = node[0]
        G.nodes[node_id]['error'] = node_errors[i].item()

    # Plotting error regions in the graph
    # plot_error_regions_in_graph(G, inputs, mode=mode)
    plot_error_clusters(G, inputs)

    print("Max error:", node_errors.max().item())
    print("Min error:", node_errors.min().item())

    threshold = 0.2  # or 0.2, or relative to the scale of your data
    errors = torch.abs(predictions - targets)
    correct = (errors < threshold).float()
    accuracy = correct.mean()  # % of predictions within the threshold
    print(f"Threshold-based accuracy: {accuracy.item() * 100:.2f}%")

    relative_errors = torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)
    accuracy = (relative_errors < 0.2).float().mean()  # within 10%
    print(f"Relative accuracy (<10% error): {accuracy.item() * 100:.2f}%")
    # print("Original depth measures shape:", depth_measures.shape)  # Should be [num_nodes]

# def test():
#     out = model(data)
#     mse = F.mse_loss(out, data.y, reduction = 'none')
#     mse = mse.mean(dim=1)  # Mean squared error for each node
#     print(f"Test MSE: {mse}")
#     print(f"Max MSE: {mse.max()}")
#     print(f"Min MSE: {mse.min()}")
#     # let's print the overall accuracy 
#     accuracy = 1 - (mse.mean() / data.y.mean())
#     print("Overall accuracy:", accuracy.item())

# with torch.no_grad():
#     model.eval()
#     test()