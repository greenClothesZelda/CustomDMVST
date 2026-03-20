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
    all_predictions = []
    all_labels = []
    
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    for batch in dataloader:
        #print(batch)
        demands_series = batch['demands_series'].to(device)
        weather = batch['weather'].to(device)
        labels = batch['labels'].to(device)
        time = batch['time'].to(device)
        sample_idx = batch['sample_idx'].to(device)
        
        outputs = model(demands_series, weather, time, sample_idx, labels=labels, return_dict=True)
        predictions = outputs['predictions']
        
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    dist = torch.abs(all_predictions - all_labels).flatten()

    num_nodes = test_dataset.dataset.num_nodes
    dropped_points = test_dataset.dataset.dropped_points
    
    mae = torch.mean(dist).item()
    mape = torch.mean(dist / (all_labels.flatten() + 1)).item() * 100

    result_df = pd.DataFrame({
        'Predictions': all_predictions.numpy().tolist(),
        'Labels': all_labels.numpy().tolist()
    })
    csv_path = output_dir / 'test_results.csv'
    result_df.to_csv(csv_path, index=False)
    visualize_predictions(csv_path, num_nodes=num_nodes, output_dir=output_dir)
    print(f'num_nodes: {num_nodes}, dropped_points: {dropped_points}')
    return {'MAE': mae, 'Origin_MAE': mae * num_nodes + dropped_points, 'MAPE': mape}


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
