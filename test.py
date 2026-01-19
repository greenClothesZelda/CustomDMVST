import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.loader.GNVSTDataset import MyGraphDataset


# =========================================================
# 1. PyG Dataset -> 일반 Dataset
# =========================================================
class TabularDataset(Dataset):
    def __init__(self, pyg_dataset, indices):
        self.dataset = pyg_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]

        # x_sum : [8]
        x_sum = data.x[:, :, 0].sum(axis=0).float()

        # weather : [8, 4]
        weather = data.weather.float()

        # y : scalar (IMPORTANT)
        y = torch.sum(data.y).float()

        return x_sum, weather, y


# =========================================================
# 2. Model
# =========================================================
class TestModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=8, output_size=1):
        super().__init__()
        self.weather_fc = nn.Linear(4, 2)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, weather):
        """
        x       : [B, 8]
        weather : [B, 8, 4]
        """
        weather_emb = self.weather_fc(weather)     # [B, 8, 2]
        x = x.unsqueeze(-1)                        # [B, 8, 1]

        emb = torch.cat([x, weather_emb], dim=-1) # [B, 8, 3]

        _, (h, _) = self.lstm(emb)
        out = self.fc(h[-1])                       # [B, 1]
        return out.squeeze(1)                      # [B]


# =========================================================
# 3. Train & Test
# =========================================================
def main():
    # -----------------------------
    # Dataset
    # -----------------------------
    pyg_dataset = MyGraphDataset(root="./data", time_step=8)

    indices = np.arange(len(pyg_dataset))
    np.random.shuffle(indices)

    train_size = int(0.8 * len(indices))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    train_dataset = TabularDataset(pyg_dataset, train_idx)
    test_dataset = TabularDataset(pyg_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # -----------------------------
    # Model / Optim
    # -----------------------------
    model = TestModel(input_size=3, hidden_size=8, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # -----------------------------
    # Train
    # -----------------------------
    model.train()
    epochs = 20

    for epoch in range(epochs):
        total_loss = 0.0

        for x_sum, weather, y in train_loader:
            pred = model(x_sum, weather)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train MSE: {avg_loss:.6f}")

    # -----------------------------
    # Test (MAE)
    # -----------------------------
    model.eval()
    total_abs_error = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_sum, weather, y in test_loader:
            pred = model(x_sum, weather)

            # safety: avoid broadcasting bugs
            pred = pred.view(-1)
            y = y.view(-1)

            total_abs_error += torch.abs(pred - y).sum().item()
            total_samples += y.numel()

    mae = total_abs_error / total_samples
    print(f"\n✅ Test MAE: {mae:.6f}")

    # -----------------------------
    # Debug sample
    # -----------------------------
    x_sum, weather, y = test_dataset[0]
    with torch.no_grad():
        pred = model(x_sum.unsqueeze(0), weather.unsqueeze(0))

    print("\n[Sample Check]")
    print("GT   :", y.item())
    print("Pred :", pred.item())


# =========================================================
if __name__ == "__main__":
    main()
