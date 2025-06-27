import torch

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data, criterion, mae_criterion):
    model.eval()
    with torch.no_grad():
        out = model(data)
        val_pred = out[data.test_mask]
        val_true = data.y[data.test_mask]
        val_loss = criterion(out[data.test_mask], data.y[data.test_mask])
        mae_loss = mae_criterion(out[data.test_mask], data.y[data.test_mask])

    return val_loss.item(), mae_loss.item(), val_pred.cpu().numpy(), val_true.cpu().numpy()