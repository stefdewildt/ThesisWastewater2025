import random
import numpy as np
from sklearn.metrics import r2_score

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data

from data.load_graph import load_graph_pickle, convert_to_pyg
from models.gcn import GCN
from models.graphsage import GraphSAGERegressor
from utils.preprocessing import split_data
from utils.train_help import train, validate
from utils.visualize import plot_training_loss

from config import *

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

G = load_graph_pickle(GRAPH_PATH)
data = convert_to_pyg(G)

input = input('Do you want to use GCN or GraphSAGE? (gcn/graphsage): ').strip().lower()
if input == 'gcn':
    model = GCN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(DEVICE)
elif input == 'graphsage':  
    model = GraphSAGERegressor(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT).to(DEVICE)
data = data.to(DEVICE)

data.train_mask, data.test_mask = split_data(data.num_nodes, TRAIN_SPLIT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()

losses = []
val_losses = []
for epoch in range(EPOCHS):
    loss = train(model, data, optimizer, criterion)
    val_loss, mae_loss, val_pred_np, val_true_np = validate(model, data, criterion, mae_criterion)
    r2 = r2_score(val_true_np, val_pred_np)

    if epoch % 20 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, MAE Loss: {mae_loss:.4f}, R2: {r2:.4f}')
    losses.append(loss)
    val_losses.append(val_loss)
    
print(f'Epoch: {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, MAE Loss: {mae_loss:.4f}, R2: {r2:.4f}')

# print(f'Validation true: {val_true_np}, Validation pred: {val_pred_np}')

torch.save(model.state_dict(), 'checkpoints/gcn_model.pt')

plot_training_loss(losses, val_losses)
