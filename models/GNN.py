import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, k, conv_in, conv_hidden, conv_out):
        super().__init__()
        
        # 활성화 함수 클래스 가져오기 (예: "ReLU")
        act_fn = getattr(nn, conv_hidden.act) 
        
        # 1. 입력 레이어
        self.conv_in = GCNConv(conv_in.in_channels, conv_hidden.hidden_channels)
        self.act_in = act_fn()

        # 2. 은닉 레이어들: nn.ModuleList를 사용하여 레이어를 올바르게 등록
        self.hiddens = nn.ModuleList([
            GCNConv(conv_hidden.hidden_channels, conv_hidden.hidden_channels)
            for _ in range(k - 2)
        ])
        self.hidden_acts = nn.ModuleList([act_fn() for _ in range(k - 2)])

        # 3. 출력 레이어
        self.conv_out = GCNConv(conv_hidden.hidden_channels, conv_out.out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        N, T, F = x.size()
        xi_list = []
        for i in range(T):
            xi = x[:, i, :]  # 시점 i의 노드 특징
            xi = self.conv_in(xi, edge_index, edge_attr)
            xi = self.act_in(xi)
            for conv, act in zip(self.hiddens, self.hidden_acts):
                xi = conv(xi, edge_index, edge_attr)
                xi = act(xi)
            xi = self.conv_out(xi, edge_index, edge_attr)
            xi_list.append(xi.unsqueeze(1))  # 시점 차원 추가
        x = torch.cat(xi_list, dim=1)  # 시점 차원에서 연결
        return x
        
