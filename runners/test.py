import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd


@torch.no_grad()
def test_loop(model, test_dataset, output_dir, device):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dataset = test_dataset.dataset
    retained_indices = base_dataset.retained_flat_indices.to(torch.long)
    total_num_points = base_dataset.total_num_points
    region_mean = base_dataset.origin_demand_arr.to(torch.float).mean(dim=0)
    all_predictions = []
    all_labels = []
    all_sample_indices = []
    
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    for batch in dataloader:
        demands_series = batch['demands_series'].to(device)
        weather = batch['weather'].to(device)
        labels = batch['labels'].to(device)
        time = batch['time'].to(device)
        sample_idx = batch['sample_idx'].to(device)
        
        outputs = model(demands_series, weather, time, sample_idx, labels=labels, return_dict=True)
        predictions = outputs['predictions']
        
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
        all_sample_indices.append(sample_idx.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_sample_indices = torch.cat(all_sample_indices, dim=0)
    dist = torch.abs(all_predictions - all_labels).flatten()
    sq_dist = torch.square(all_predictions - all_labels).flatten()

    num_nodes = base_dataset.num_nodes
    
    mae = torch.mean(dist).item()
    rmse = torch.sqrt(torch.mean(sq_dist)).item()
    mape = torch.mean(dist / (all_labels.flatten() + 1)).item() * 100
    full_labels = base_dataset.get_full_labels(all_sample_indices).to(torch.float)
    full_predictions = region_mean.unsqueeze(0).repeat(all_predictions.shape[0], 1)
    full_predictions = full_predictions.to(all_predictions.dtype)
    full_predictions[:, retained_indices] = all_predictions
    origin_dist = torch.abs(full_predictions - full_labels).flatten()
    origin_sq_dist = torch.square(full_predictions - full_labels).flatten()
    origin_rmse = torch.sqrt(torch.mean(origin_sq_dist)).item()
    origin_mape = torch.mean(origin_dist / (full_labels.flatten() + 1)).item() * 100

    result_df = pd.DataFrame({
        'Predictions': all_predictions.numpy().tolist(),
        'Labels': all_labels.numpy().tolist()
    })
    csv_path = output_dir / 'test_results.csv'
    result_df.to_csv(csv_path, index=False)
    visualize_predictions(csv_path, num_nodes=num_nodes, output_dir=output_dir)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Origin_RMSE': origin_rmse, 'Origin_MAPE': origin_mape}


def visualize_predictions(csv_path, num_nodes, output_dir):
    df = pd.read_csv(csv_path)
    pred = np.stack(df['Predictions'].apply(ast.literal_eval).map(np.asarray).to_list())
    labels = np.stack(df['Labels'].apply(ast.literal_eval).map(np.asarray).to_list())

    pred = pred.reshape(-1, num_nodes)
    labels = labels.reshape(-1, num_nodes)
    diff = np.abs(labels - pred)

    demand_mean = np.sum(labels, axis=1) / num_nodes
    error_mean = np.mean(diff, axis=1)

    plt.figure(figsize=(24, 4))
    plt.plot(demand_mean, label='Average Demand', color='blue')
    plt.plot(error_mean, label='Mean Absolute Error', color='green')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Demand and Prediction Error Over Time')
    plt.legend()
    plt.savefig(output_dir / 'demand_error_analysis.png')
    plt.close()

    node_demands = np.sum(labels, axis=0)
    max_demand_node = int(np.argmax(node_demands))
    min_demand_node = int(np.argmin(node_demands))
    mid_demand_node = int(np.argsort(node_demands)[num_nodes // 2])

    visualize_sample(pred[:, max_demand_node], labels[:, max_demand_node], output_dir, name='max_demand_node')
    visualize_sample(pred[:, min_demand_node], labels[:, min_demand_node], output_dir, name='min_demand_node')
    visualize_sample(pred[:, mid_demand_node], labels[:, mid_demand_node], output_dir, name='mid_demand_node')


def visualize_sample(pred, labels, output_dir, name):
    plt.figure(figsize=(24, 4))
    plt.plot(labels, label='Labels', color='blue')
    plt.plot(pred, label='Predictions', color='orange')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    plt.title(f'Predictions vs Labels for Sample Node: {name}')
    plt.legend()
    plt.savefig(output_dir / f'predictions_{name}.png')
    plt.close()
