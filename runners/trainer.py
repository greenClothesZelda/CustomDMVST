import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
import copy

log = logging.getLogger(__name__)


def train_one_epoch(model, loader, device, optimizer, criterion_cluster, criterion_node, alpha, scheduler):
    model.train()
    total_loss = 0
    for (node_data, cluster_data) in loader:
        node_data = node_data.to(device)
        cluster_data = cluster_data.to(device)
        optimizer.zero_grad()
        cluster_out, node_out = model(
            node_data=node_data, cluster_data=cluster_data, context_data=None)
        # print(f'cluster_out shape: {cluster_out.shape}, cluster_data.y shape: {cluster_data.y.shape}')
        # print(f'node_out shape: {node_out.shape}, node_data.y shape: {node_data.y.shape}')
        loss_cluster = criterion_cluster(
            cluster_out.reshape(-1, 1), cluster_data.y.view(-1, 1).to(device))
        loss_node = criterion_node(node_out.reshape(-1, 1),
                              node_data.y.view(-1, 1).to(device))
        loss = alpha * loss_cluster + (1-alpha) * loss_node
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, device, criterion_cluster, criterion_node, alpha):
    model.eval()
    total_loss = 0
    for (node_data, cluster_data) in loader:
        node_data = node_data.to(device)
        cluster_data = cluster_data.to(device)
        cluster_out, node_out = model(
            node_data, cluster_data, context_data=None)
        loss_cluster = criterion_cluster(
            cluster_out.reshape(-1, 1), cluster_data.y.view(-1, 1).to(device))
        loss_node = criterion_node(node_out.reshape(-1, 1), node_data.y.view(-1, 1).to(device))
        loss = alpha * loss_cluster + (1-alpha) * loss_node
        total_loss += loss.item()
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / len(loader)
    return avg_loss


def train(model, train_loader, val_loader, device, optimizer, criterion_cluster, criterion_node, scheduler, epochs, alpha, patience=10):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None

    wait = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, device, optimizer, criterion_cluster, criterion_node, alpha, scheduler)
        val_loss = validate(model, val_loader, device,
                            criterion_cluster, criterion_node, alpha)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if scheduler is not None:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            log.info(
                f"New best model found at epoch {epoch+1} with val loss {val_loss:.4f}")

        log.info(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # break

        # 조기종료
        if epoch > 10 and val_loss > best_val_loss:
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
    total_samples = 0
    all_targets = []
    for (node_data, cluster_data) in test_loader:
        node_data = node_data.to(device)
        cluster_data = cluster_data.to(device)
        cluster_out, node_out = model(
            node_data, cluster_data, context_data=None)
        node_out = node_out.reshape(-1, 1)
        all_outputs.append(node_out.cpu())
        all_targets.append(node_data.y.view(-1, 1).cpu())
        loss = criterion(node_out.reshape(-1, 1), node_data.y.view(-1, 1).to(device))
        total_loss += loss.item() * node_data.y.size(0)
        total_samples += node_data.y.size(0)
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / total_samples * max_demand
    covered_loss = avg_loss

    if coverage:
        covered_loss = avg_loss * coverage
        log.info(
            f"Coverage applied: {coverage}, Covered Test Loss: {covered_loss:.4f}")
    else:
        log.info("No coverage information provided.")
    all_outputs = np.round(
        torch.cat(all_outputs, dim=0).squeeze().numpy() * max_demand)
    all_targets = torch.cat(all_targets, dim=0).squeeze().numpy() * max_demand

    # print(f'All outputs shape: {all_outputs.shape}, All targets shape: {all_targets.shape}')
    results_df = pd.DataFrame({
        'Predicted': all_outputs,
        'Actual': all_targets
    })
    results_path = f"{save_root}/test_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Test results saved to {results_path}")
    return avg_loss, covered_loss
