import torch
import random

def split_data(num_nodes, train_ratio=0.8):
    indices = list(range(num_nodes))
    random.shuffle(indices)
    train_size = int(train_ratio * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    return train_mask, test_mask