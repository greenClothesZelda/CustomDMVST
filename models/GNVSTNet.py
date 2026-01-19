import torch
import torch.nn as nn

from models.GNN import GNN
from models.loader.GNVSTDataset import MyGraphDataset
from torch_geometric.loader import DataLoader


class GNVSTNet(nn.Module):
    def __init__(
            self,
            context_dim,
            temporal_data_size,
            Node_GNN_configs,
            Node_LSTM_configs,
            aggregated_LSTM_configs,
            Meteorological_configs,
            device
    ):
        super().__init__()

        self.node_gnn = GNN(**Node_GNN_configs)

        self.aggregated_lstm = nn.LSTM(
            input_size= 1 + Meteorological_configs['out_features'],
            batch_first=True,
            **aggregated_LSTM_configs
        )

        self.node_lstm = nn.LSTM(
            input_size= 1 + #Node_GNN_configs['conv_out']['out_channels'] +
            Meteorological_configs['out_features'],
            batch_first=True,
            **Node_LSTM_configs
        )

        self.aggregated_final_out = nn.Linear(
            aggregated_LSTM_configs['hidden_size']+context_dim,
            1
        )

        self.node_final_out = nn.Linear(
            Node_LSTM_configs['hidden_size']+context_dim ,
            1
        )

        self.temporal_embedding_layer = nn.Linear(  # meteorological data를 임베딩하는 선형층
            in_features=Meteorological_configs['in_features'],  # 날씨 피처 개수
            out_features=Meteorological_configs['out_features']
        )
        self.sigmoid = nn.Sigmoid()

        with torch.no_grad():
            self.node_final_out.weight.fill_(0.0)
            self.node_final_out.bias.fill_(0.0)
            
            self.aggregated_final_out.weight.fill_(0.0)
            self.aggregated_final_out.bias.fill_(0.0)

            print("Initialized final linear layers with zeros.")
            print(f"Node final out weights: {self.node_final_out.weight}")
            print(f'Node final out bias: {self.node_final_out.bias}')


    def forward(
            self,
            node_data,
            context_data=None
    ):
        num_graphs = node_data.num_graphs
        B = num_graphs
        x = node_data.x[:, :, 0]  # [B*n, time_step]
        T = x.size(1)

        weather = node_data.weather  # [B * time_step, 4]
        weather_embedded = self.temporal_embedding_layer(weather)  # [B * time_step, temporal_data_size]
        #print(f'weather_embedded shape: {weather_embedded.shape}')
        aggregated_x = node_data.all_x.view(B, -1) 
        mean_aggregated_x = torch.mean(aggregated_x, dim=1)
        aggregated_x = torch.cat([aggregated_x.unsqueeze(-1), weather_embedded.view(B, -1, weather_embedded.size(-1))], dim=-1) # [B, n, 1 + temporal_data_size]
        aggregated_x, _ = self.aggregated_lstm(aggregated_x)
        aggregated_x = aggregated_x[:, -1, :]  # [B, lstm_hidden_size]
        aggregated_x = self.aggregated_final_out(aggregated_x).squeeze(-1) + mean_aggregated_x

        mean_x = torch.mean(x, dim=1) # [B, n]
        #print(f'x shape: {x.shape}, weather_embedded shape: {weather_embedded.shape}')
        x = x.view(B, -1, T, 1)
        weather_embedded= weather_embedded.view(B, 1, T, -1).expand(-1, x.size(1), -1, -1)
        x = torch.cat([x, weather_embedded], dim=-1)  # [B*n, time_step, node_feature + temporal_data_size]
        x = x.view(B * x.size(1), T, -1)  # [B*n, time_step, node_feature + temporal_data_size]
        #print(f'x before GNN shape: {x.shape}')
        x, _ = self.node_lstm(x)
        x = x[:, -1, :]  # [B*n, lstm_hidden_size]
        x = self.node_final_out(x).squeeze(-1)

        return aggregated_x, x, mean_x