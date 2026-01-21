import torch
import pandas as pd
import json


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, time_step, target_columns=['강수량(mm)', '기온(°C)', '습도(%)', '적설(cm)']):
        super(MyDataset, self).__init__()
        with open(root/'gwn_data.json', 'r') as f:
            demand_data = json.load(f)

        df = pd.read_csv(root/'meteorological_data.csv', encoding='cp949')
        df_filled = df.fillna(0)
        target_columns = target_columns
        #     # "풍속(m/s)",
        #     "강수량(mm)",
        #     "기온(°C)",
        #     "습도(%)",
        #     # "일조(hr)",
        #     "적설(cm)",
        #     # "전운량(10분위)",
        #     # "현지기압(hPa)"
        weather = df_filled[target_columns].values
        self.weather_list = torch.tensor(weather, dtype=torch.float)
        self.time = torch.eye(7).unsqueeze(1).unsqueeze(0) # (1, 7, 1, 7)
        self.time = self.time.repeat(weather.shape[0]//(7*24), 1, 24, 1).view(-1,7)  # (num_samples, 7)

        time = torch.arange(0, 24).unsqueeze(1).repeat(self.time.shape[0]//24, 1).view(-1,1) / 24.0  # (num_samples, 1)
        self.time = torch.cat([self.time, time], dim=1)  # (num_samples, 8)

        self.demand_arr = torch.tensor(demand_data['x'], dtype=torch.long)
        
        self.num_nodes = self.demand_arr.shape[1]
        self.time_step = time_step
        self.max_demand = int(torch.max(self.demand_arr).item())

        self.dropped_points = demand_data['meta']['dropped_points']

    def __getitem__(self, index):
        return {
            'demands_series': self.demand_arr[index:index + self.time_step].transpose(0, 1),  # (N, T)
            'labels': self.demand_arr[index + self.time_step],
            'time': self.time[index + self.time_step],
            'weather': self.weather_list[index + self.time_step]
        }

    def __len__(self):
        return self.demand_arr.shape[0] - self.time_step