import os
import torch
from torch_geometric.data import InMemoryDataset, Data
import json

class MyGraphDataset(InMemoryDataset):
    def __init__(
            self, 
            root, 
            time_step,
            transform=None, 
            pre_transform=None,
            max_demand=12
            ):
        # root는 데이터가 저장될 경로입니다.
        self.time_step = time_step
        self.max_demand = max_demand
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], 
            weights_only=False  # 이 부분을 추가하세요!
        )

    @property
    def raw_file_names(self):
        # 원본 파일들의 이름을 리스트로 반환 (있으면 체크, 없으면 download 실행)
        return ['gwn_data.json']

    @property
    def processed_file_names(self):
        # 처리가 완료된 후 저장될 파일 이름
        return ['gwn_data.pt']

    def download(self):
        # 원본 데이터를 다운로드하는 로직 (이미 있다면 pass)
        pass

    def process_edges(self, edges_list):
        # 1. edge_index 추출: [[u1, u2, ...], [v1, v2, ...]] 형태
        sources = [e['u'] for e in edges_list]
        targets = [e['v'] for e in edges_list]
        edge_index = torch.tensor([sources, targets], dtype=torch.long)

        # 2. edge_attr 추출: [[w1], [w2], ...] 형태 (보통 [E, 1] 모양으로 만듭니다)
        weights = [[e['w']] for e in edges_list]
        edge_attr = torch.tensor(weights, dtype=torch.float)

        return edge_index, edge_attr
    
    def process(self):
        # [핵심] 원본 데이터를 읽어 Data 객체 리스트로 변환하는 곳
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        print(json_data.keys())
        print(json_data['edges'][:5])
        edge_index, edge_attr = self.process_edges(json_data['edges']) 

        data_list = []

        all_features = []
        for idx, sample in enumerate(json_data['meta']['hours']):
            all_features.append(torch.tensor(json_data['x'][idx], dtype=torch.float).div(self.max_demand))  # 정규화
        
        for i in range(len(all_features) - self.time_step):
            window_x = torch.stack(all_features[i : i + self.time_step])  # (time_step, num_nodes, feature_dim)
            #print(window_x.shape)
            x = window_x.permute(1, 0).unsqueeze(-1)  # (num_nodes, time_step, 1) feature가 1이라서 unsqueeze
            y = all_features[i + self.time_step]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, time=torch.tensor(i, dtype=torch.long))
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])