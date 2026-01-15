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
            Cluster_GNN_configs,
            Node_GNN_configs,
            LSTM_configs,
            assignment_matrix,
            device
    ):
        super().__init__()
        self.assignment_matrix = assignment_matrix.to(device)  # [num_nodes, num_clusters]

        self.cluster_gnn = GNN(**Cluster_GNN_configs)

        self.node_gnn = GNN(**Node_GNN_configs)

        self.cluster_lstm = nn.LSTM(
            input_size=Cluster_GNN_configs['conv_out']['out_channels'] +
            temporal_data_size,
            **LSTM_configs
        )

        self.node_lstm = nn.LSTM(
            input_size=Node_GNN_configs['conv_out']['out_channels'] +
            temporal_data_size,
            **LSTM_configs
        )

        self.cluster_final_out = nn.Linear(
            LSTM_configs['hidden_size']+context_dim,
            1
        )

        self.node_final_out = nn.Linear(
            LSTM_configs['hidden_size']*2+context_dim ,
            1
        )

        self.temporal_embedding_layer = nn.Linear(  # meteorological data를 임베딩하는 선형층
            in_features=4,  # 날씨 피처 개수
            out_features=temporal_data_size
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            node_data,
            cluster_data,
            context_data=None
    ):
        B = node_data.num_graphs
        N = node_data.x.size(0) // B
        T = node_data.x.size(1)
        C = self.assignment_matrix.size(1)
        # 그래프 처리
        # [num_nodes*batch, time_step, gnn_out_features]
        x = self.node_gnn(node_data) # [num_nodes*batch, time_step, gnn_out_features]

        node_x = x.view(B, T, N, -1)
        cluster_sum = torch.einsum('n c, b t n f -> b t c f',
                                   self.assignment_matrix, node_x)
        node_count = self.assignment_matrix.sum(dim=0)  # [num_clusters]
        node_count = torch.clamp(node_count, min=1e-9)

        cluster_mean = cluster_sum / node_count.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        cluster_mean = cluster_mean.view(B, T, -1, cluster_mean.size(-1))
        cluster_mean = cluster_mean.permute(0, 2, 1, 3).contiguous().view(-1, T, cluster_mean.size(-1))
        # print(f'cluster_mean shape: {cluster_mean.shape}')
        cluster_data.x = torch.cat([cluster_data.x, cluster_mean], dim=-1)
        cluster_out = self.cluster_gnn(cluster_data)

        # temporal data 처리
        weather = node_data.weather  # [time_step*batch, weather_features]
        weather_embedded = self.temporal_embedding_layer(weather)  # 선형층 통과

        # LSTM 입력 준비
        weather_embedded = weather_embedded.view(B, T, -1)
        # [num_nodes*batch, time_step, weather_embedded_features]
        weather_cluster_embedded = weather_embedded[cluster_data.batch]
        # print(weather_embedded.shape)

        # [num_nodes*batch, time_step, gnn_out + weather_embedded_features]
        cluster_lstm_input = torch.cat([cluster_out, weather_cluster_embedded], dim=-1)
        # print(lstm_input.shape)

        _, (cluster_lstm_out, _) = self.cluster_lstm(cluster_lstm_input)

        weather_node_embedded = weather_embedded[node_data.batch]

        node_lstm_input = torch.cat([x, weather_node_embedded], dim=-1)
        _, (node_lstm_out, _) = self.node_lstm(node_lstm_input)

        # print(f'cluster_lstm_out shape: {cluster_lstm_out.shape}')
        # print(f'node_lstm_out shape: {node_lstm_out.shape}')

        cluster_lstm_out = cluster_lstm_out.view(B, C, -1)
        node_lstm_out = node_lstm_out.view(B, N, -1)

        #cluster -> node로 매핑
        cluster2node = torch.einsum('n c, b c f -> b n f',
                                     self.assignment_matrix, cluster_lstm_out)
        cluster_count = self.assignment_matrix.sum(dim=1)  # [num_nodes]
        cluster_count = torch.clamp(cluster_count, min=1e-9)
        cluster2node = cluster2node / cluster_count.unsqueeze(0).unsqueeze(-1)
        # print(f'cluster2node shape: {cluster2node.shape}')
        # print(f'node_lstm_out shape before concat: {node_lstm_out.shape}')
        node_lstm_out = torch.cat([node_lstm_out, cluster2node], dim=-1)
        # print(f'node_lstm_out shape after concat: {node_lstm_out.shape}')

        cluster_final = self.cluster_final_out(cluster_lstm_out) 
        node_final = self.node_final_out(
            node_lstm_out
        )
        return self.sigmoid(cluster_final), node_final  # 0~1사이 값 반환