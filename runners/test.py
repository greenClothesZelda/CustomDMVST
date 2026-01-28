
import torch
import pandas as pd

import test


@torch.no_grad()
def test_loop(model, test_dataset, output_dir, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    for batch in dataloader:
        #print(batch)
        demands_series = batch['demands_series'].to(device)
        weather = batch['weather'].to(device)
        labels = batch['labels'].to(device)
        time = batch['time'].to(device)
        
        outputs = model(demands_series, weather, time, labels=labels, return_dict=True)
        predictions = outputs['predictions']
        
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    dist = torch.abs(all_predictions - all_labels).flatten()

    num_nodes = test_dataset.dataset.num_nodes
    dropped_points = test_dataset.dataset.dropped_points
    
    mae = torch.mean(dist).item()
    mape = torch.mean(dist / (all_labels.flatten() + 1)).item()

    result = pd.DataFrame({
        'Predictions': all_predictions.numpy().flatten(),
        'Labels': all_labels.numpy().flatten()
    })
    result.to_csv(f'{output_dir}/test_results.csv', index=False)
    print(f'num_nodes: {num_nodes}, dropped_points: {dropped_points}')
    return {'MAE': mae, 'Origin_MAE': mae * num_nodes + dropped_points, 'MAPE': mape}