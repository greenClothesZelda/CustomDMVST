import torch
import torch.nn as nn

class ModelTrainer(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = kwargs.get('loss', nn.L1Loss())
        self.relu = nn.ReLU()

    def forward(self, demands_series, weather, time, sample_idx, labels=None, return_dict=True, **kwargs):
        predictions = self.model(demands_series, weather, time, sample_idx)

        loss = None
        if labels is not None:
            loss = self.loss_fn(predictions, labels.to(torch.float32))
        if return_dict:
            predictions = self.relu(predictions)
            return {
                'predictions': predictions,
                'loss': loss
            }
        else:
            return predictions, loss


def collate_fn(features):
    batch = {}
    keys = features[0].keys()
    for key in keys:
        batch[key] = torch.stack([feature[key] for feature in features], dim=0)
    return batch
