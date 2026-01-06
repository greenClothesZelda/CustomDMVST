import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
import copy

log = logging.getLogger(__name__)

def train_one_epoch(model, loader, device, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch, context_data=None)
        loss = criterion(out, batch.y.view(-1, 1).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / len(loader)
    return avg_loss

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, context_data=None)
        loss = criterion(out, batch.y.view(-1, 1).to(device))
        total_loss += loss.item()
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / len(loader)
    return avg_loss

def train(model, train_loader, val_loader, device, optimizer, criterion, scheduler, epochs):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None

    patience = 10
    wait = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, scheduler)
        val_loss = validate(model, val_loader, device, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            log.info(f"New best model found at epoch {epoch+1} with val loss {val_loss:.4f}")

        log.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # break

        #조기종료
        if epoch > 10 and val_loss >= best_val_loss:
            wait += 1
            if wait >= patience:
                log.info(f'Early stopping at epoch {epoch+1}')
                break
        else:
            wait = 0
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log.info(f'Best model state loaded loss: {best_val_loss:.4f}')
    return train_losses, val_losses

@torch.no_grad()
def test(model, test_loader, device, save_root, max_demand, coverage=None):
    model.eval()
    all_outputs = []
    criterion = nn.L1Loss()
    total_loss = 0
    all_targets = []
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch, context_data=None)
        all_outputs.append(out.cpu())
        all_targets.append(batch.y.view(-1, 1).cpu())
        loss = criterion(out, batch.y.view(-1, 1).to(device))
        total_loss += loss.item()
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / len(test_loader) * max_demand
    covered_loss = avg_loss
    
    if coverage:
        covered_loss = avg_loss * coverage
        log.info(f"Coverage applied: {coverage}, Covered Test Loss: {covered_loss:.4f}")
    else:
        log.info("No coverage information provided.")
    all_outputs = np.round(torch.cat(all_outputs, dim=0).squeeze().numpy() * test_loader.dataset.max_demand)
    all_targets = torch.cat(all_targets, dim=0).squeeze().numpy()  * test_loader.dataset.max_demand
    results_df = pd.DataFrame({
        'Predicted': all_outputs,
        'Actual': all_targets
    })
    results_path = f"{save_root}/test_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Test results saved to {results_path}")
    return avg_loss, covered_loss

        