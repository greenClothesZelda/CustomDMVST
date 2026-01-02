import torch
import logging

log = logging.getLogger(__name__)

def train_one_epoch(model, loader, temporal_dataset, device, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch, temporal_data=temporal_dataset[batch.time] if temporal_dataset is not None else None, context_data=None)
        loss = criterion(out, batch.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

@torch.no_grad()
def validate(model, loader, temporal_dataset, device, criterion):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, temporal_data=temporal_dataset[batch.time] if temporal_dataset is not None else None, context_data=None)
        loss = criterion(out, batch.y.view(-1, 1).to(device))
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

def train(model, train_loader, val_loader, temporal_dataset, device, optimizer, criterion, scheduler, epochs):
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, temporal_dataset, device, optimizer, criterion, scheduler)
        val_loss = validate(model, val_loader, temporal_dataset, device, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step()

        log.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        #조기종료
        if epoch > 10 and val_losses[-1] > val_losses[-2] > val_losses[-3]:
            log.info(f"Early stopping triggered. Epoch: {epoch+1}")
            break
    return train_losses, val_losses

@torch.no_grad()
def test(model, test_loader, temporal_dataset, device, save_root):
    import pandas as pd
    import os
    
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch, temporal_data=temporal_dataset[batch.time] if temporal_dataset is not None else None, context_data=None)

        out.multiply_(12.0)  # 필요시 스케일링 조정 TODO
        batch.y.multiply_(12.0)  # 필요시 스케일링 조정 TODO

        preds = out.cpu().numpy().flatten()
        labels = batch.y.cpu().numpy().flatten()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # Save to CSV
    df = pd.DataFrame({
        'prediction': all_preds,
        'label': all_labels
    })
    os.makedirs(save_root, exist_ok=True)
    csv_path = os.path.join(save_root, 'predictions.csv')
    df.to_csv(csv_path, index=False)
    log.info(f"Predictions saved to {csv_path}")
    
    # Calculate accuracy with rounding
    rounded_preds = [round(p) for p in all_preds]
    correct = sum(1 for pred, label in zip(rounded_preds, all_labels) if pred == label)
    accuracy = correct / len(all_labels)
    
    log.info(f"Test Accuracy: {accuracy:.4f}")
    
    return accuracy