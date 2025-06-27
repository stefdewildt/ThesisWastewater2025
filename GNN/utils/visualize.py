import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import DBSCAN
from scipy.interpolate import griddata



def plot_training_loss(losses, val_losses): 
    plt.plot(losses, label = 'training loss')
    plt.plot(val_losses, label = 'validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    # plt.savefig('../img/neural_network/training_loss_GS_5.png')
    plt.show()

# def plot_error_regions_in_graph(G, input):
#     # Extract error values
#     error_dict = nx.get_node_attributes(G, 'error')

#     # Filter nodes that have both position and error
#     pos = {
#         node: (attrs["x"], attrs["y"])
#         for node, attrs in G.nodes(data=True)
#         if node in error_dict and "x" in attrs and "y" in attrs
#     }

#     nodes_to_draw = list(pos.keys())
#     errors = np.array([error_dict[node] for node in nodes_to_draw])

#     # Normalize and colormap
#     norm = plt.Normalize(errors.min(), errors.max())
#     cmap = plt.cm.coolwarm
#     normalized_colors = cmap(norm(errors))

#     # Create plot
#     fig, ax = plt.subplots(figsize=(10, 10))
#     nx.draw(
#         G.subgraph(nodes_to_draw),
#         pos=pos,
#         node_size=10,
#         node_color=normalized_colors,
#         with_labels=False,
#         ax=ax
#     )

#     # Add colorbar
#     sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])
#     fig.colorbar(sm, ax=ax, label='Error (MSE)')

#     # plt.savefig('../img/neural_network/depth_error_02.png', dpi=300)
#     plt.title(f'Node-wise MSE Errors with {input.upper()} Model')
#     plt.savefig(f'../img/error_maps/depth_error_{input}_02.png', dpi=300)
#     plt.show()

def plot_error_regions_in_graph(G, input, mode="nodes"):
    """
    Plot error distribution on a graph.
    
    Parameters:
        G (networkx.Graph): The graph with 'x', 'y', and 'error' node attributes.
        input (str): Label for the model used.
        mode (str): 'nodes', 'clusters', or 'heatmap'
    """
    error_dict = nx.get_node_attributes(G, 'error')

    # Filter nodes that have both position and error
    pos = {
        node: (attrs["x"], attrs["y"])
        for node, attrs in G.nodes(data=True)
        if node in error_dict and "x" in attrs and "y" in attrs
    }

    if not pos:
        raise ValueError("No nodes with both position and error attributes found.")

    nodes_to_draw = list(pos.keys())
    errors = np.array([error_dict[node] for node in nodes_to_draw])
    x_vals = [pos[n][0] for n in nodes_to_draw]
    y_vals = [pos[n][1] for n in nodes_to_draw]

    fig, ax = plt.subplots(figsize=(10, 10))

    # NODE-WISE COLORING
    norm = plt.Normalize(errors.min(), errors.max())
    cmap = plt.cm.coolwarm
    node_colors = cmap(norm(errors))

    nx.draw(
        G.subgraph(nodes_to_draw),
        pos=pos,
        node_size=10,
        node_color=node_colors,
        with_labels=False,
        ax=ax
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Error (MSE)')

    plt.title(f'Node-wise MSE Errors with {input.upper()} Model')

    # plt.savefig(f'../img/error_maps/depth_error_{input}_{mode}.png', dpi=300)
    plt.show()

def plot_error_clusters(G, input):

    error_dict = nx.get_node_attributes(G, 'error')

    # Filter nodes that have both position and error
    pos = {
        node: (attrs["x"], attrs["y"])
        for node, attrs in G.nodes(data=True)
        if node in error_dict and "x" in attrs and "y" in attrs
    }

    threshold = 0.25
    high_error_nodes = [
        node for node in G.nodes()
        if 'error' in G.nodes[node] and G.nodes[node]['error'] > threshold
    ] 

    # Step 2: Get their spatial positions
    high_error_coords = np.array([
        (G.nodes[node]['x'], G.nodes[node]['y']) for node in high_error_nodes
    ])
    # DBSCAN CLUSTERING
    clustering = DBSCAN(eps=50, min_samples=3).fit(high_error_coords)
    labels = clustering.labels_

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab10(labels / labels.max() if labels.max() > 0 else 0)

    # Plot original graph in gray
    nx.draw(G, 
        pos=pos, 
        node_color='lightgray', 
        node_size=10, 
        with_labels=False, 
        ax=ax)
    
    for i, node in enumerate(high_error_nodes):
        x, y = G.nodes[node]['x'], G.nodes[node]['y']
        cluster_label = labels[i]
        if cluster_label != -1:  # -1 means noise
            ax.scatter(x, y, color=colors[i], s=30, edgecolor='black', zorder=5)
            None
    

    plt.title(f'Error Clusters (DBSCAN) with {input.upper()} Model')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.savefig(f'../img/clusters/depth_error_clusters_{input}_03_25p.png', dpi=300)

    plt.show()

# def plot_heatmap(G, input):
#     # INTERPOLATED HEATMAP
#     grid_x, grid_y = np.mgrid[
#         min(x_vals):max(x_vals):200j,
#         min(y_vals):max(y_vals):200j
#     ]
#     grid_z = griddata((x_vals, y_vals), errors, (grid_x, grid_y), method='cubic')

#     heat = ax.imshow(
#         grid_z.T,
#         extent=(min(x_vals), max(x_vals), min(y_vals), max(y_vals)),
#         origin='lower',
#         cmap='coolwarm',
#         alpha=0.7
#     )
#     ax.scatter(x_vals, y_vals, c='black', s=2, label='Nodes')
#     fig.colorbar(heat, ax=ax, label='Interpolated Error')
#     plt.title(f'Interpolated Error Heatmap with {input.upper()} Model')

#     pass