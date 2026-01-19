import torch
import torch.nn as nn

class IRModule(nn.Module):
    def __init__(self, IRdataset):
        super().__init__()
        if hasattr(IRdataset, 'dataset'):
            source_dataset = IRdataset.dataset
        else:
            source_dataset = IRdataset

        self.num_nodes = source_dataset.num_nodes
        self.max_demand = source_dataset.max_demand
        self.num_weather_features = source_dataset.weather_list.shape[1]
    def forward(self, x):
        pass

class IRVSTNet(nn.Module):
    def __init__(self, ir_module, embedding_dim, **kwargs):
        super().__init__()
        self.ir_module = ir_module

        self.node_embedding = nn.Embedding(
            num_embeddings=ir_module.num_nodes,
            embedding_dim=embedding_dim
        )

        self.demand_embedding = nn.Embedding(
            num_embeddings=ir_module.max_demand + 1,
            embedding_dim=embedding_dim
        )

        self.weather_embedding_layer = nn.Linear(
            in_features=ir_module.num_weather_features,
            out_features=embedding_dim
        )

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=embedding_dim,
            hidden_size=kwargs['lstm']['hidden_size'],
            num_layers=kwargs['lstm']['num_layers'],
            dropout=kwargs['lstm']['dropout'],
            bidirectional=True
        )
        self.lstm_embedding_layer = nn.Linear(
            in_features=2 * kwargs['lstm']['hidden_size'],
            out_features=embedding_dim
        )

        self.final_layer = nn.Sequential(
            nn.Linear(
                in_features=embedding_dim,
                out_features=1
            )
        )

    def demand_average(self, demands_series):
        demands_series = demands_series.to(torch.float32)
        return torch.mean(demands_series, dim=2, keepdim=True) # (B, N, 1)

    def forward(self, demands_series, weather):
        B, N, T = demands_series.size()
        embedded_demands = self.demand_embedding(
            demands_series
        )  # (B, N, T, D)

        weather_embedded = self.weather_embedding_layer(weather).unsqueeze(1) # (B, 1, D)
        weather_embedded = weather_embedded.unsqueeze(2).expand(-1, N, T, -1)  # (B, N, T, D)

        node_indices = torch.arange(N, device=demands_series.device).unsqueeze(0).expand(B, N)
        node_embedded = self.node_embedding(node_indices)  # (B, N, D)
        node_embedded = node_embedded.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, D)

        lstm_input = embedded_demands + node_embedded + weather_embedded  # (B, N, T, D)
        lstm_input = lstm_input.view(B * N, T, -1)  # (B*N, T, D)

        lstm_out, _ = self.lstm(lstm_input)  # (B*N, T, 2*H)
        lstm_out = self.lstm_embedding_layer(lstm_out[:, -1, :])  # (B*N, D)
        lstm_out = lstm_out.view(B, N, -1)  # (B, N, D)

        #TODO: IR 모듈과의 연동

        out = self.final_layer(lstm_out)  # (B, N, 1)
        out += self.demand_average(demands_series)  # (B, N, 1)
        return out.squeeze(-1)  # (B, N)
    
class ModelTrainer(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = kwargs.get('loss', nn.L1Loss())

    def forward(self, demands_series, weather, labels=None, return_dict=True, **kwargs):
        predictions = self.model(demands_series, weather)

        loss = None
        if labels is not None:
            loss = self.loss_fn(predictions, labels)
        if return_dict:
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
        batch[key] = torch.stack([f[key] for f in features], dim=0)
    return batch