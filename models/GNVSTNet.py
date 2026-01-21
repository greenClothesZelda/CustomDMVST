import torch
import torch.nn as nn

class IRModule(nn.Module):
    def __init__(self, IRdataset, device, k):
        super().__init__()
        self.device = device
        self.k = k
        assert k > 0, "k must be greater than 0"
        if hasattr(IRdataset, 'dataset'):
            source_dataset = IRdataset.dataset
        else:
            source_dataset = IRdataset

        self.num_nodes = source_dataset.num_nodes
        self.max_demand = source_dataset.max_demand
        self.num_weather_features = source_dataset.weather_list.shape[1]

        x_list = []
        y_list = []

        for data in IRdataset:
            #print(f'Data x shape: {data["demands_series"].shape}, y shape: {data["labels"].shape}')
            x_list.append(data['demands_series'].to(torch.float32)) # (N, T)
            y_list.append(data['labels'].unsqueeze(1)) # (N, 1)

        self.db_keys = torch.stack(x_list).permute(1, 0, 2).to(device) # (N, Samples, T)
        self.db_values = torch.stack(y_list).permute(1, 0, 2).to(device) # (N, Samples, 1)

        self.num_samples = self.db_keys.shape[1]

        print(f'IRModule initialized: Nodes={self.num_nodes}, Samples={self.num_samples}, Time_Steps={self.db_keys.shape[2]}')
        print(f'db_keys shape: {self.db_keys.shape}, db_values shape: {self.db_values.shape}')

        self.db_norms = torch.norm(self.db_keys, dim=2, keepdim=True) + 1e-8
        
    def forward(self, query):        
        queries = query.to(self.device).unsqueeze(2).to(torch.float32) # (B, N, 1, T)
        db_keys_transposed = self.db_keys.permute(0, 2, 1) # (N, T, Samples)

        dot = torch.matmul(queries, db_keys_transposed) # (B, N, 1, Samples)

        q_norm = torch.norm(queries, dim=3, keepdim=True) + 1e-8 # (B, N, 1, 1)
        db_norms = self.db_norms.permute(0, 2, 1) # (N, 1, Samples)
        norms = q_norm * db_norms # (B, N, 1, Samples)

        cosine_similarities = dot / norms # (B, N, 1, Samples)
        value, indices = torch.topk(cosine_similarities, self.k, dim=3) # (B, N, 1, k)
        weight = torch.softmax(value, dim=3) # (B, N, 1, k)

        db_v = self.db_values.unsqueeze(0)
        idx = indices.transpose(2,3) # (B, N, k, 1)

        retrieved_values = torch.gather(
            db_v.expand(queries.size(0), -1, -1, -1), 
            2, 
            idx
        )

        aggregated = torch.sum(retrieved_values.squeeze(-1) * weight.squeeze(2), dim=2)
        return aggregated, idx # (B, N)
    
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
        B, N, T, D = x.size()
        x_reshaped = x.view(B * T, N, D)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.view(B, T, N, D)
        lstm_input = attn_out.reshape(B * N, T, D)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = lstm_out.view(B, N, -1)
        return lstm_out # (B, N, 2*hidden_dim)

class IRVSTNet(nn.Module):
    def __init__(self, ir_module, embedding_dim, **kwargs):
        super().__init__()
        self.ir_module = ir_module
        for param in self.ir_module.parameters():
            param.requires_grad = False

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

        # self.lstm = nn.LSTM(
        #     batch_first=True,
        #     input_size=embedding_dim,
        #     hidden_size=kwargs['lstm']['hidden_size'],
        #     num_layers=kwargs['lstm']['num_layers'],
        #     dropout=kwargs['lstm']['dropout'],
        #     bidirectional=True
        # )
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
            in_features=embedding_dim+1,
            out_features=1
        )

        self.time_embedding_layer = nn.Linear(
            in_features=8,
            out_features=embedding_dim
        )
    def demand_average(self, demands_series):
        demands_series = demands_series.to(torch.float32)
        return torch.mean(demands_series, dim=2, keepdim=True) # (B, N, 1)

    def forward(self, demands_series, weather, time):
        B, N, T = demands_series.size()
        embedded_demands = self.demand_embedding(
            demands_series
        )  # (B, N, T, D)

        weather_embedded = self.weather_embedding_layer(weather).unsqueeze(1) # (B, 1, D)
        weather_embedded = weather_embedded.unsqueeze(2).expand(-1, N, T, -1)  # (B, N, T, D)

        time_embedded = self.time_embedding_layer(time).unsqueeze(1)  # (B, 1, D)
        time_embedded = time_embedded.unsqueeze(2).expand(-1, N, T, -1)  # (B, N, T, D)

        node_indices = torch.arange(N, device=demands_series.device).unsqueeze(0).expand(B, N)
        node_embedded = self.node_embedding(node_indices)  # (B, N, D)
        node_embedded = node_embedded.unsqueeze(2).expand(-1, -1, T, -1)  # (B, N, T, D)

        lstm_input = embedded_demands + node_embedded + weather_embedded + time_embedded  # (B, N, T, D)

        lstm_out = self.attn_lstm(lstm_input)  # (B*N, T, 2*H)
        lstm_out = self.lstm_embedding_layer(lstm_out)  # (B*N, D)
        lstm_out = lstm_out.view(B, N, -1)  # (B, N, D)

        ir_out, _ = self.ir_module(demands_series)  # (B, N)

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

    def forward(self, demands_series, weather, time, labels=None, return_dict=True, **kwargs):
        predictions = self.model(demands_series, weather, time)

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
        batch[key] = torch.stack([f[key] for f in features], dim=0)
    return batch