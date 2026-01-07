import os
from xml.sax.handler import all_features
import torch
from torch_geometric.data import InMemoryDataset, Data
import json
import logging
import pandas as pd
log = logging.getLogger(__name__)

class ClusterDataset(InMemoryDataset):
    def __init__(
            self, 
            root, 
            time_step,
            transform=None, 
            pre_transform=None
            ):
        self.time_step = time_step
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], 
            weights_only=False
        )

        stats = torch.load(self.processed_paths[1])
        self.max_demand = stats['max_demand']
        self.count_data = stats['count_data']
        self.assignment_matrix = torch.load(self.processed_paths[2])
        log.info(f"Max cluster demand: {self.max_demand}"
                 f", Count data sample: {list(self.count_data.items())}")

    @property
    def raw_file_names(self):
        return ['gwn_data.json', 'meteorological_data.csv']
    
    @property
    def processed_file_names(self):
        return [f'cluster_data_{self.time_step}.pt', f'cluster_stats_{self.time_step}.pt', f'assignments_.pt']
    
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
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        edge_index, edge_attr = self.process_edges(json_data['clusters_edges']) 
        data_list = []

        max_demand = max(map(max, json_data['clusters_demands']))

        all_demands = []
        count_data = {}
        total_clusters = len(json_data['clusters_demands'][0])
        for idx, time in enumerate(json_data['meta']['hours']):
            demand_t = torch.tensor(json_data['clusters_demands'][idx], dtype=torch.float).div(max_demand)
            one_hot_t = torch.eye(total_clusters)
            combined_features = torch.cat([demand_t.unsqueeze(1), one_hot_t], dim=1)
            all_demands.append(combined_features)

            # 각 값의 빈도 카운트
            for val in json_data['clusters_demands'][idx]:
                count_data[val] = count_data.get(val, 0) + 1

        #기상데이터 처리
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[1]), encoding="cp949")
        df_filled = df.fillna(0)
        target_columns = [
            #"풍속(m/s)",
            "강수량(mm)",
            "기온(°C)",
            "습도(%)",
            #"일조(hr)",
            "적설(cm)",
            #"전운량(10분위)",
            #"현지기압(hPa)"
        ]
        weather = df_filled[target_columns].values
        self.weather_list = torch.tensor(weather, dtype=torch.float)

        for i in range(len(all_demands) - self.time_step):
            x = torch.stack(all_demands[i:i + self.time_step], dim=0)  # [time_step, num_nodes, features]
            x = x.permute(1, 0, 2)  # [num_nodes, time_step, features]
            y = all_demands[i + self.time_step][:, 0]  # [num_nodes]

            weather = self.weather_list[i: i + self.time_step]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, weather=weather)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(
            {
                'max_demand': max_demand,
                'count_data': count_data,
            }, 
            self.processed_paths[1]
        )
        torch.save(
            torch.tensor(json_data['cluster_assignment'], dtype=torch.float), 
            self.processed_paths[2]
        )