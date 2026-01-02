import torch
import torch.nn as nn

from models.GNN import GNN

class GNVSTNet(nn.Module):
    def __init__(
            self,
            context_dim,
            temporal_data_size,
            GNN_configs,
            LSTM_configs,
    ):
        super().__init__()
        
        self.gnn = GNN(**GNN_configs)

        self.lstm = nn.LSTM(
            input_size = GNN_configs['conv_out']['out_channels'] + temporal_data_size,
            **LSTM_configs
        )

        self.final_out = nn.Linear(
            LSTM_configs['hidden_size']+context_dim,
            1
        )

        self.temporal_embedding_layer = nn.Linear( #meteorological data를 임베딩하는 선형층
            in_features=8,
            out_features=temporal_data_size
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(
            self,
            data,
            temporal_data = None,
            context_data = None
    ):
        gnn_out = self.gnn(data.x, data.edge_index) # (N, T, gnn_hidden)
        gnn_out = gnn_out.permute(1, 0, 2)  # LSTM 입력을 위해 (T, N, gnn_hidden)
        
        temporal_data = temporal_data.to(gnn_out.device) if temporal_data is not None else torch.zeros(gnn_out.size(0), 8).to(gnn_out.device)  # (T, 8)
        temporal_embedded = self.temporal_embedding_layer(temporal_data)  # (T, temporal_data_size)
        temporal_embedded = temporal_embedded.unsqueeze(1).expand(-1, gnn_out.size(1), -1)  # (T, N, temporal_data_size)

        lstm_input = torch.cat([gnn_out, temporal_embedded], dim=-1)  # (T, N, gnn_hidden + temporal_data_size)
        lstm_input = lstm_input.permute(1, 0, 2)  # (N, T, gnn_hidden + temporal_data_size)
        _, (lstm_out, _) = self.lstm(lstm_input)
        lstm_out = lstm_out.squeeze(0)  # (N, hidden_size)

        context_data = context_data.to(gnn_out.device) if context_data is not None else torch.zeros(gnn_out.size(1), 0).to(gnn_out.device)  # (N, context_dim)
        final_input = torch.cat([lstm_out, context_data], dim=-1)  # (N, hidden_size + context_dim)
        gnn_out = self.final_out(final_input)  # (N, 1)
        gnn_out = self.sigmoid(gnn_out)  # (N, 1)
        return gnn_out



