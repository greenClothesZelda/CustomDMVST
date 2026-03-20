import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class IRModule(nn.Module):
    def __init__(self, dataset, device, k):
        super().__init__()
        self.device = device
        self.k = k
        if k <= 0:
            raise ValueError("k must be greater than 0")

        if hasattr(dataset, 'dataset'):
            source_dataset = dataset.dataset
        else:
            source_dataset = dataset

        self.num_nodes = source_dataset.num_nodes
        self.max_demand = source_dataset.max_demand
        self.num_weather_features = source_dataset.weather_list.shape[1]

        key_list = []
        value_list = []

        for data in dataset:
            key_list.append(data['demands_series'].to(torch.float32))
            value_list.append(data['labels'].unsqueeze(1).to(torch.float32))

        self.db_keys = torch.stack(key_list).permute(1, 0, 2).to(device)  # (N, Samples, T)
        self.db_values = torch.stack(value_list).permute(1, 0, 2).to(device)  # (N, Samples, 1)
        self.db_norms = torch.norm(self.db_keys, dim=2) + 1e-8  # (N, Samples)
        self.num_samples = self.db_keys.shape[1]

        log.info(
            "IRModule initialized: Nodes=%s, Samples=%s, Time_Steps=%s",
            self.num_nodes,
            self.num_samples,
            self.db_keys.shape[2]
        )

    def forward(self, query_demands, sample_idx):
        queries = query_demands.to(self.device).to(torch.float32)  # (B, N, T)
        sample_idx = sample_idx.to(self.device).to(torch.long)
        batch_size, num_nodes, _ = queries.size()

        aggregated = torch.zeros(batch_size, num_nodes, device=self.device, dtype=torch.float32)
        retrieved_indices = torch.full(
            (batch_size, num_nodes, self.k),
            fill_value=-1,
            device=self.device,
            dtype=torch.long
        )

        for batch_idx in range(batch_size):
            candidate_count = int(sample_idx[batch_idx].item())
            top_k = min(self.k, candidate_count)
            if top_k == 0:
                continue

            query = queries[batch_idx]  # (N, T)
            prefix_keys = self.db_keys[:, :candidate_count, :]  # (N, candidate_count, T)
            prefix_values = self.db_values[:, :candidate_count, 0]  # (N, candidate_count)
            prefix_norms = self.db_norms[:, :candidate_count]  # (N, candidate_count)

            dot = torch.matmul(query.unsqueeze(1), prefix_keys.transpose(1, 2)).squeeze(1)  # (N, candidate_count)
            query_norm = torch.norm(query, dim=1, keepdim=True) + 1e-8  # (N, 1)
            cosine_similarities = dot / (query_norm * prefix_norms)

            values, indices = torch.topk(cosine_similarities, top_k, dim=1)
            weights = torch.softmax(values, dim=1)
            retrieved_values = torch.gather(prefix_values, 1, indices)

            aggregated[batch_idx] = torch.sum(retrieved_values * weights, dim=1)
            retrieved_indices[batch_idx, :, :top_k] = indices

        return aggregated, retrieved_indices


class SpatialAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes, num_layers, nhead, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.num_nodes = num_nodes

    def forward(self, x):
        batch_size, num_nodes, time_step, embedding_dim = x.size()
        x_reshaped = x.view(batch_size * time_step, num_nodes, embedding_dim)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.view(batch_size, time_step, num_nodes, embedding_dim)
        lstm_input = attn_out.reshape(batch_size * num_nodes, time_step, embedding_dim)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = lstm_out.view(batch_size, num_nodes, -1)
        return lstm_out  # (B, N, 2*hidden_dim)


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

        self.attn_lstm = SpatialAttentionLSTM(
            input_dim=embedding_dim,
            hidden_dim=kwargs['lstm']['hidden_size'],
            num_nodes=ir_module.num_nodes,
            num_layers=kwargs['lstm']['num_layers'],
            nhead=kwargs['lstm']['nhead'],
            dropout=kwargs['lstm']['dropout']
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
        self.lambda_layer = nn.Linear(
            in_features=embedding_dim + 1,
            out_features=1
        )

        self.time_embedding_layer = nn.Linear(
            in_features=8,
            out_features=embedding_dim
        )

    def forward(self, demands_series, weather, time, sample_idx):
        batch_size, num_nodes, time_step = demands_series.size()
        embedded_demands = self.demand_embedding(
            demands_series.long()
        )  # (B, N, T, D)

        weather_embedded = self.weather_embedding_layer(weather.to(torch.float32)).unsqueeze(1)
        weather_embedded = weather_embedded.unsqueeze(2).expand(-1, num_nodes, time_step, -1)  # (B, N, T, D)

        time_embedded = self.time_embedding_layer(time.to(torch.float32)).unsqueeze(1)
        time_embedded = time_embedded.unsqueeze(2).expand(-1, num_nodes, time_step, -1)  # (B, N, T, D)

        node_indices = torch.arange(num_nodes, device=demands_series.device).unsqueeze(0).expand(batch_size, num_nodes)
        node_embedded = self.node_embedding(node_indices)
        node_embedded = node_embedded.unsqueeze(2).expand(-1, -1, time_step, -1)  # (B, N, T, D)

        lstm_input = embedded_demands + node_embedded + weather_embedded + time_embedded  # (B, N, T, D)

        lstm_out = self.attn_lstm(lstm_input)
        lstm_out = self.lstm_embedding_layer(lstm_out)
        lstm_out = lstm_out.view(batch_size, num_nodes, -1)  # (B, N, D)

        ir_out, _ = self.ir_module(demands_series, sample_idx)  # (B, N)

        out = self.final_layer(lstm_out).squeeze(-1)  # (B, N)

        lambda_input = torch.cat([lstm_out, ir_out.unsqueeze(-1)], dim=-1)  # (B, N, D+1)
        lambda_weight = torch.sigmoid(self.lambda_layer(lambda_input)).squeeze(-1)
        out = lambda_weight * out + (1 - lambda_weight) * ir_out  # (B, N)

        return out  # (B, N)


class ModelTrainer(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = kwargs.get('loss', nn.L1Loss())

    def forward(self, demands_series, weather, time, sample_idx, labels=None, return_dict=True, **kwargs):
        predictions = self.model(demands_series, weather, time, sample_idx)

        loss = None
        if labels is not None:
            loss = self.loss_fn(predictions, labels.to(torch.float32))
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
        batch[key] = torch.stack([feature[key] for feature in features], dim=0)
    return batch
