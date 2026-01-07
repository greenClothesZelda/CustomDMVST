import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from models.loader.GNVSTDataset import MyGraphDataset
from models.loader.ClusterDataset import ClusterDataset

class JointDataset(Dataset):
    def __init__(self, node_dataset: MyGraphDataset, cluster_dataset: ClusterDataset):
        super().__init__()
        self.node_dataset = node_dataset
        self.cluster_dataset = cluster_dataset

        self.assignment_matrix = cluster_dataset.assignment_matrix

        # 두 데이터셋의 길이가 같아야 함을 확인
        assert len(node_dataset) == len(cluster_dataset), "두 데이터셋의 길이는 같아야 합니다."

    def len(self):
        return len(self.node_dataset)

    def get(self, idx):
        # 동일한 인덱스 t에 대해 두 데이터를 튜플 형태로 반환
        data_a = self.node_dataset[idx]
        data_b = self.cluster_dataset[idx]
        return data_a, data_b
    
def get_joint_datasets(node_dataset_config, cluster_dataset_config):
    node_dataset = MyGraphDataset(**node_dataset_config)
    cluster_dataset = ClusterDataset(**cluster_dataset_config)
    joint_dataset = JointDataset(node_dataset=node_dataset, cluster_dataset=cluster_dataset)
    return joint_dataset