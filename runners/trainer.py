import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
import torch.nn.functional as F

log = logging.getLogger(__name__)


def train_one_epoch(model, loader, device, optimizer, criterion_cluster, criterion_node, alpha, scheduler):
    model.train()
    total_loss = 0
    for (node_data, cluster_data) in loader:
        node_data = node_data.to(device)
        cluster_data = cluster_data.to(device)
        optimizer.zero_grad()
        cluster_out, node_out, masking = model(
            node_data=node_data, cluster_data=cluster_data, context_data=None)
        # print(f'cluster_out shape: {cluster_out.shape}, cluster_data.y shape: {cluster_data.y.shape}')
        # print(f'node_out shape: {node_out.shape}, node_data.y shape: {node_data.y.shape}')
        loss_cluster = criterion_cluster(
            cluster_out.reshape(-1, 1), cluster_data.y.view(-1, 1).to(device))
        B = node_data.num_graphs
        loss_node = criterion_node(node_out.reshape(B, -1),
                                   node_data.y.view(B, -1).to(device))
        
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
        cluster_out, node_out, masking = model(
            node_data, cluster_data, context_data=None)
        loss_cluster = criterion_cluster(
            cluster_out.reshape(-1, 1), cluster_data.y.view(-1, 1).to(device))
        
        B = node_data.num_graphs
        loss_node = criterion_node(
            node_out.reshape(B, -1), node_data.y.view(B, -1).to(device))
        loss = alpha * loss_cluster + (1-alpha) * loss_node
        total_loss += loss.item()
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / len(loader)
    return avg_loss


def train(model, train_loader, val_loader, device, optimizer, criterion_cluster, criterion_node, scheduler, epochs, alpha, patience=10, min_delta=1e-4):
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

        # 개선 여부 확인 (min_delta 적용)
        # 이전 최고 기록보다 min_delta 이상으로 줄어들었을 때만 개선으로 인정
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            wait = 0  # 개선되었으므로 대기 카운트 초기화
            log.info(
                f"New best model found at epoch {epoch+1} with val loss {val_loss:.4f}")
        else:
            wait += 1
            log.info(f"EarlyStopping counter: {wait} out of {patience}")

        log.info(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 조기 종료 체크
        if wait >= patience:
            log.info(
                f'Early stopping at epoch {epoch+1} due to insufficient improvement.')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        log.info(f'Best model state loaded loss: {best_val_loss:.4f}')

    return train_losses, val_losses


@torch.no_grad()
def test(model, test_loader, device, save_root, max_demand, assignment_matrix, dropped_point=0):
    model.eval()
    assignment_matrix = assignment_matrix.to(device)
    all_outputs = []
    criterion = nn.L1Loss()
    total_loss = 0
    total_samples = 0
    all_targets = []
    for (node_data, cluster_data) in test_loader:
        node_data = node_data.to(device)
        cluster_data = cluster_data.to(device)
        cluster_out, node_out, masking = model(
            node_data, cluster_data, context_data=None)
        B = node_data.num_graphs
        node_out = node_out.reshape(B, -1)
        node_distribution = F.softmax((assignment_matrix.unsqueeze(0) * node_out.unsqueeze(-1)).permute(0, 2, 1), dim=-1)  # [B, num_clusters, num_nodes]
        cluster_y = cluster_data.y.view(B, -1)

        node_allocations = torch.sum(assignment_matrix, dim=1)  # [num_clusters]
        node_allocations = torch.clamp(node_allocations, min=1e-9)

        # print(f'node_distribution shape: {node_distribution.shape}, cluster_y shape: {cluster_y.shape}, node_allocations shape: {node_allocations.shape}')

        reconstructed_node_y = torch.einsum('b c, b c n -> b n',
                                            cluster_y, node_distribution)
        reconstructed_node_y = reconstructed_node_y / node_allocations.unsqueeze(0)
        loss = criterion(reconstructed_node_y, node_data.y.view(B, -1))
        total_loss += loss.item() * B
        total_samples += B
        all_outputs.append(reconstructed_node_y.reshape(-1).cpu())
        all_targets.append(node_data.y.cpu())
        
    avg_loss = total_loss / total_samples * max_demand
    log.info(f"Test Loss: {avg_loss:.4f}")
    log.info(f"Dropped Points during testing: {dropped_point}")
    original_loss = avg_loss * assignment_matrix.size(0) + dropped_point

    all_outputs = torch.cat(all_outputs, dim=0).squeeze().numpy() * max_demand
    all_targets = torch.cat(all_targets, dim=0).squeeze().numpy() * max_demand

    print(f'All outputs shape: {all_outputs.shape}, All targets shape: {all_targets.shape}')
    results_df = pd.DataFrame({
        'Predicted': all_outputs,
        'Actual': all_targets
    })
    results_path = f"{save_root}/test_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Test results saved to {results_path}")
    return avg_loss, original_loss
