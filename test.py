import torch
import torch.nn.functional as F
import torch.nn as nn

from pathlib import Path

if __name__ == "__main__":
    t1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t2 = torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.5, 0.6]])

    t1_dist = F.softmax(t1, dim=-1)
    t2_dist = F.softmax(t2, dim=-1)

    print("t1 distribution:", t1_dist)
    print("t2 distribution:", t2_dist)

    cse = nn.CrossEntropyLoss()
    loss = cse(t1_dist, t2_dist)
    print("Cross-Entropy Loss:", loss.item())