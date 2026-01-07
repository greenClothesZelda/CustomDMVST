from omegaconf import OmegaConf
from models import GNVSTNet
from models.loader.GNVSTDataset import MyGraphDataset
from models.loader.ClusterDataset import ClusterDataset
from models.loader.jointDataset import JointDataset, get_joint_datasets
from torch_geometric.loader import DataLoader
import torch

from pathlib import Path

if __name__ == "__main__":
    output_path = Path("outputs/2026-01-07/15-54-38")
    config_path = output_path / '.hydra' / 'config.yaml'
    config = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(config))

    dataset = get_joint_datasets(
        node_dataset_config=config.dataset, cluster_dataset_config=config.dataset)
    
    model = GNVSTNet(
        context_dim=config.model.context_dim,
        temporal_data_size=config.model.temporal_data_size,
        Node_GNN_configs=config.model.Node_GNN,
        Cluster_GNN_configs=config.model.Cluster_GNN,
        LSTM_configs=config.model.LSTM,
        assignment_matrix=dataset.assignment_matrix,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    checkpoint_path = output_path / 'final_model.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print("Model loaded successfully.")
    test_dataset = dataset[int(len(dataset)*(config.loader.train.ratio + config.loader.val.ratio)):]
    test_loader = DataLoader(
        test_dataset, batch_size=config.loader.test.batch_size, shuffle=config.loader.test.shuffle)
    
    


