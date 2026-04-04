from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.loader.DemandDataset import MyDataset


def _write_synthetic_dataset(root: Path, timesteps: int, height: int, width: int):
    grid = np.repeat(
        np.arange(timesteps, dtype=np.int64)[:, None, None],
        height,
        axis=1,
    )
    grid = np.repeat(grid, width, axis=2)
    np.save(root / "grid(1).npy", grid.astype(np.int32))

    timestamps = pd.date_range("2025-01-01 00:00", periods=timesteps, freq="H")
    weather = pd.DataFrame(
        {
            "일시": timestamps.strftime("%Y-%m-%d %H:%M"),
            "강수량(mm)": np.linspace(0, 1, timesteps),
            "기온(°C)": np.linspace(10, 20, timesteps),
            "습도(%)": np.linspace(40, 50, timesteps),
            "적설(cm)": np.zeros(timesteps),
        }
    )
    weather.to_csv(root / "meteorological_data.csv", index=False, encoding="cp949")


def test_dataset_builds_stresnet_keyframes(tmp_path):
    _write_synthetic_dataset(tmp_path, timesteps=20, height=2, width=3)

    dataset = MyDataset(
        root=tmp_path,
        size=1,
        train_ratio=0.5,
        len_c=3,
        len_p=1,
        len_t=1,
        period_interval=4,
        trend_interval=8,
    )

    assert dataset.base_offset == 8
    assert len(dataset) == 12

    item = dataset[0]
    assert item["demands_series"].shape == (5, 2, 3)
    assert item["labels"].shape == (6,)
    assert item["sample_idx"].item() == 0

    assert torch.all(item["demands_series"][0] == 7)
    assert torch.all(item["demands_series"][1] == 6)
    assert torch.all(item["demands_series"][2] == 5)
    assert torch.all(item["demands_series"][3] == 4)
    assert torch.all(item["demands_series"][4] == 0)
    assert torch.all(item["labels"] == 8)

    full_labels = dataset.get_full_labels(torch.tensor([0, 1]))
    assert full_labels.shape == (2, 6)
    assert torch.all(full_labels[0] == 8)
    assert torch.all(full_labels[1] == 9)
