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
            in_features=4,  #날씨 피처 개수
            out_features=temporal_data_size
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(
            self,
            data,
            context_data = None
    ):
        B = data.num_graphs
        N = data.x.size(0) // B
        T = data.x.size(1)
        #그래프 처리
        x = self.gnn(data)  # [num_nodes*batch, time_step, gnn_out_features]

        #temporal data 처리
        weather = data.weather  # [time_step*batch, weather_features]
        weather_embedded = self.temporal_embedding_layer(weather)  # 선형층 통과

        #LSTM 입력 준비
        weather_embedded = weather_embedded.view(B, T, -1)
        weather_embedded = weather_embedded[data.batch] # [num_nodes*batch, time_step, weather_embedded_features]
        # print(weather_embedded.shape)

        lstm_input = torch.cat([x, weather_embedded], dim=-1)  # [num_nodes*batch, time_step, gnn_out + weather_embedded_features]
        # print(lstm_input.shape)

        _, (out, _) = self.lstm(lstm_input) 

        # print(out.shape)
        out = self.final_out(out.squeeze(0))  # [num_nodes*batch, 1]
        return self.sigmoid(out)

        
        
        
        

        



