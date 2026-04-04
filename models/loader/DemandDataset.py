import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        size,
        train_ratio,
        len_c,
        len_p,
        len_t,
        period_interval,
        trend_interval,
        target_columns=None,
    ):
        super().__init__()
        self.root = root
        self.len_c = len_c
        self.len_p = len_p
        self.len_t = len_t
        self.period_interval = period_interval
        self.trend_interval = trend_interval
        self.base_offset = max(
            self.len_c,
            self.len_p * self.period_interval,
            self.len_t * self.trend_interval,
        )

        if target_columns is None:
            target_columns = ["강수량(mm)", "기온(°C)", "습도(%)", "적설(cm)"]

        weather_df = pd.read_csv(root / "meteorological_data.csv", encoding="cp949").fillna(0)
        timestamps = pd.to_datetime(weather_df["일시"])
        weekday = torch.tensor(timestamps.dt.dayofweek.to_numpy(), dtype=torch.long)
        weekday_one_hot = F.one_hot(weekday, num_classes=7).to(torch.float32)
        hour_feature = torch.tensor(
            timestamps.dt.hour.to_numpy(),
            dtype=torch.float32,
        ).unsqueeze(1) / 24.0
        self.time = torch.cat([weekday_one_hot, hour_feature], dim=1)

        raw_weather = torch.tensor(
            weather_df[target_columns].to_numpy(),
            dtype=torch.float32,
        )

        grid = np.load(root / f"grid({size}).npy")
        if grid.ndim == 4 and grid.shape[1] == 1:
            grid = grid[:, 0]
        if grid.ndim != 3:
            raise ValueError(f"Expected grid(shape) to be (T, H, W), got {grid.shape}.")

        self.grid = torch.from_numpy(grid).to(torch.long)
        self.num_timesteps, self.grid_H, self.grid_W = self.grid.shape

        if raw_weather.shape[0] != self.num_timesteps:
            raise ValueError(
                "meteorological_data.csv and grid timeline length must match: "
                f"{raw_weather.shape[0]} vs {self.num_timesteps}"
            )

        self.origin_demand_arr = self.grid.reshape(self.num_timesteps, -1)
        self.total_num_points = self.origin_demand_arr.shape[1]
        self.num_nodes = self.total_num_points
        self.retained_flat_indices = torch.arange(self.total_num_points, dtype=torch.long)
        self.max_demand = int(self.origin_demand_arr.max().item())

        self.num_samples = self.num_timesteps - self.base_offset
        if self.num_samples <= 0:
            raise ValueError(
                f"Not enough timesteps ({self.num_timesteps}) for base_offset ({self.base_offset})."
            )

        train_sample_count = int(self.num_samples * train_ratio)
        if train_sample_count <= 0:
            raise ValueError("train_ratio is too small for the available dataset length.")

        train_target_end = self.base_offset + train_sample_count
        weather_train = raw_weather[self.base_offset:train_target_end]
        weather_mean = weather_train.mean(dim=0)
        weather_std = weather_train.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.weather_mean = weather_mean
        self.weather_std = weather_std
        self.weather_list = (raw_weather - weather_mean) / weather_std

        log.info(
            "Dataset initialized: grid=%s, total_points=%s, base_offset=%s, samples=%s, "
            "weather_dim=%s",
            tuple(self.grid.shape),
            self.total_num_points,
            self.base_offset,
            self.num_samples,
            self.weather_list.shape[1],
        )

    def __getitem__(self, index):
        target_idx = index + self.base_offset
        frames = []

        for step in range(1, self.len_c + 1):
            frames.append(self.grid[target_idx - step])
        for step in range(1, self.len_p + 1):
            frames.append(self.grid[target_idx - step * self.period_interval])
        for step in range(1, self.len_t + 1):
            frames.append(self.grid[target_idx - step * self.trend_interval])

        demands_series = torch.stack(frames, dim=0)

        return {
            "demands_series": demands_series,
            "labels": self.origin_demand_arr[target_idx],
            "time": self.time[target_idx],
            "weather": self.weather_list[target_idx],
            "sample_idx": torch.tensor(index, dtype=torch.long),
        }

    def __len__(self):
        return self.num_samples

    def get_full_label(self, sample_idx):
        target_idx = int(sample_idx) + self.base_offset
        return self.origin_demand_arr[target_idx]

    def get_full_labels(self, sample_indices):
        sample_indices = torch.as_tensor(sample_indices, dtype=torch.long)
        target_indices = sample_indices + self.base_offset
        return self.origin_demand_arr[target_indices]
