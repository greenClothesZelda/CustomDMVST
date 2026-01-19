import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
import torch.nn.functional as F

log = logging.getLogger(__name__)


def train_one_epoch(model, loader, device, optimizer, criterion_cluster, criterion_node, alpha):
    model.train()
    total_loss = 0
    for node_data in loader:
        node_data = node_data.to(device)
        optimizer.zero_grad()
        aggregated_out, node_out, mean_x = model(
            node_data=node_data, context_data=None)
        node_out = node_out + mean_x
        loss_cluster = criterion_cluster(
            aggregated_out.reshape(-1, 1), node_data.all_y.view(-1, 1).to(device))
        B = node_data.num_graphs
        loss_node = criterion_node(
            node_out.reshape(B, -1), node_data.y.view(B, -1).to(device))
        loss = alpha * loss_cluster + (1 - alpha) * loss_node
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
    for node_data in loader:
        node_data = node_data.to(device)
        aggregated_out, node_out, mean_x = model(node_data, context_data=None)
        node_out = node_out + mean_x
        loss_cluster = criterion_cluster(
            aggregated_out.reshape(-1, 1), node_data.all_y.view(-1, 1).to(device))
        B = node_data.num_graphs
        loss_node = criterion_node(
            node_out.reshape(B, -1), node_data.y.view(B, -1).to(device))
        loss = alpha * loss_cluster + (1 - alpha) * loss_node
        total_loss += loss.item()
        # break  # 디버깅용: 한 배치만 처리
    avg_loss = total_loss / len(loader)
    return avg_loss


def train(model, train_loader, val_loader, device, optimizer, criterion_cluster, criterion_node, scheduler, epochs, alpha, patience=10, min_delta=0):
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None

    wait = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, device, optimizer, criterion_cluster, criterion_node, alpha)
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
    all_means = []
    all_outputs = []
    moving_avg_losses = []
    criterion = nn.L1Loss()
    total_loss = 0
    total_moving_avg_loss = 0
    total_samples = 0
    all_targets = []
    num_nodes = None
    for node_data in test_loader:
        node_data = node_data.to(device)
        _, node_out, mean_x = model(node_data, context_data=None)
        node_out = node_out + mean_x

        all_means.append(mean_x.cpu())
        moving_avg_loss = criterion(mean_x.view(-1, 1), node_data.y.view(-1, 1).to(device))
        moving_avg_losses.append(moving_avg_loss.item())

        B = node_data.num_graphs
        pred_nodes = node_out.reshape(B, -1)
        target_nodes = node_data.y.view(B, -1)

        if num_nodes is None:
            num_nodes = target_nodes.size(1)

        loss = criterion(pred_nodes, target_nodes)
        total_loss += loss.item() * B
        total_moving_avg_loss += moving_avg_loss.item() * B
        total_samples += B
        all_outputs.append(pred_nodes.reshape(-1).cpu())
        all_targets.append(target_nodes.reshape(-1).cpu())

    avg_loss = total_loss / total_samples * max_demand
    moving_avg_loss = total_moving_avg_loss / total_samples * max_demand

    log.info(f"Test MAE (nodes only): {avg_loss:.4f}")
    log.info(f"Dropped Points during testing: {dropped_point}")
    original_loss = avg_loss * \
        (num_nodes if num_nodes is not None else 0) + dropped_point
    
    original_moving_avg_loss = moving_avg_loss * \
        (num_nodes if num_nodes is not None else 0) + dropped_point
    log.info(f'original moving average loss: {original_moving_avg_loss:.4f}')

    all_outputs = torch.cat(all_outputs, dim=0).squeeze().numpy() * max_demand
    all_targets = torch.cat(all_targets, dim=0).squeeze().numpy() * max_demand
    all_means = torch.cat(all_means, dim=0).squeeze().numpy() * max_demand

    print(
        f'All outputs shape: {all_outputs.shape}, All targets shape: {all_targets.shape}')
    results_df = pd.DataFrame({
        'Predicted': all_outputs,
        'Actual': all_targets,
        'Mean':  all_means,
        'Model_Effect': all_outputs - all_means
    })
    results_path = f"{save_root}/test_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"Test results saved to {results_path}")
    return avg_loss, original_loss
