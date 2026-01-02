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

    def forward(self, x, edge_index):
        N, T, F = x.size()

        # (N, T, F) -> (T, N, F)
        x = x.transpose(0, 1)
        
        outputs = []
        for t in range(T):
            # 각 타임스텝별로 GNN 적용 (동일한 가중치 공유)
            xt = x[t] # (N, F)
            xt = self.act_in(self.conv_in(xt, edge_index))
            
            for conv, act in zip(self.hiddens, self.hidden_acts):
                xt = act(conv(xt, edge_index))
                
            xt = self.conv_out(xt, edge_index) # (N, hidden)
            outputs.append(xt)

        # 다시 합치기 (T, N, hidden) -> (N, T, hidden)
        x = torch.stack(outputs, dim=1) 
        return x # 이제 LSTM의 입력으로 들어갈 준비가 끝남! (156, 4, hidden)