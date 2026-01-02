import torch
from torch.utils.data import Dataset
import pandas as pd

class TemporalDataset(Dataset):
    def __init__(self, root, time_step=4):
        df = pd.read_csv(root, encoding="cp949")
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
        data = df_filled[target_columns].values
        self.time_step = time_step
        self.data_list = torch.tensor(data, dtype=torch.float)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx: idx + self.time_step]