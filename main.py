import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import logging

from models.GNVSTNet import GNVSTNet
from models.loader.GNVSTDataset import MyGraphDataset
from torch_geometric.loader import DataLoader
from runners.trainer import *
from models.loader.jointDataset import JointDataset, get_joint_datasets

from omegaconf import OmegaConf


log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용


class WeightedMAELoss(nn.Module):
    def __init__(self, count_data, max_demand, max_weight=30):
        super().__init__()
        self.max_demand = max_demand
        self.max_weight = max_weight

        # 전체 데이터 개수
        total_count = sum(count_data.values())

        # 각 값에 대한 가중치 계산: 100 / (해당 값의 비율 * 100)
        self.weights = {}
        for val, count in count_data.items():
            percentage = (count / total_count) * 100
            weight = min(100 / percentage, max_weight)  # 상한 100
            self.weights[val] = weight

        log.info(
            f"WeightedMAELoss initialized with {len(self.weights)} weight classes")
        log.info(f"Sample weights: {dict(list(self.weights.items())[:10])}")

    def forward(self, pred, target):
        """
        pred: [batch_size, 1] - 정규화된 예측값
        target: [batch_size, 1] - 정규화된 실제값
        """
        # 원래 스케일로 복원
        pred_orig = (pred * self.max_demand).clamp(min=0)
        target_orig = (target * self.max_demand).clamp(min=0)

        # 타겟의 정수값으로 가중치 가져오기
        target_int = torch.round(target_orig).long().squeeze()

        # 각 샘플의 가중치 가져오기
        weights = torch.tensor([
            self.weights.get(int(t.item()), self.max_weight)
            for t in target_int
        ], device=pred.device, dtype=torch.float)

        mae = torch.abs(pred_orig - target_orig).squeeze()

        # 가중 MSE
        weighted_mae = mae * weights

        # 가중 평균
        loss = weighted_mae.sum() / weights.sum()

        return loss


@hydra.main(config_path="configs", version_base=None)
def run(config):
    device = torch.device(config.device)
    log.info(f"Using device: {device}")
    dataset_cfg = config.dataset  # 모델에 입력하는 방법은 unpacking(**)사용
    model_cfg = config.model
    train_cfg = config.train

    OmegaConf.set_struct(model_cfg, False)

    # 데이터셋 및 데이터로더 설정
    dataset = get_joint_datasets(
        node_dataset_config=dataset_cfg, cluster_dataset_config=dataset_cfg)

    model_cfg['Node_GNN']['conv_in']['in_channels'] = dataset.node_dataset.num_node_features
    model_cfg['Cluster_GNN']['conv_in']['in_channels'] = dataset.cluster_dataset.num_node_features + \
        model_cfg.Node_GNN['conv_out']['out_channels']

    train_size = int(len(dataset) * config.loader.train.ratio)
    val_size = int(len(dataset) * config.loader.val.ratio)
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]

    train_loader = DataLoader(
        train_dataset, batch_size=config.loader.train.batch_size, shuffle=config.loader.train.shuffle)
    val_loader = DataLoader(
        val_dataset, batch_size=config.loader.val.batch_size, shuffle=config.loader.val.shuffle)
    test_loader = DataLoader(
        test_dataset, batch_size=config.loader.test.batch_size, shuffle=config.loader.test.shuffle)
    # 로더 설정 끝

    # 모델 초기화
    model = GNVSTNet(
        context_dim=0,
        temporal_data_size=model_cfg.temporal_data_size,
        Node_GNN_configs=model_cfg.Node_GNN,
        Cluster_GNN_configs=model_cfg.Cluster_GNN,
        LSTM_configs=model_cfg.LSTM,
        assignment_matrix=dataset.assignment_matrix,
        device=device
    ).to(device)

    # 훈련 시작
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=getattr(torch.optim, train_cfg.optimizer.type)(
            model.parameters(), **train_cfg.optimizer.params),
        criterion_cluster=WeightedMAELoss(
            count_data=dataset.cluster_dataset.count_data,
            max_demand=dataset.cluster_dataset.max_demand,
            max_weight=train_cfg.criterion.cluster_max_weight
        ),
        criterion_node=WeightedMAELoss(
            count_data=dataset.node_dataset.count_data,
            max_demand=dataset.node_dataset.max_demand,
            max_weight=train_cfg.criterion.node_max_weight
        ),
        scheduler=getattr(torch.optim.lr_scheduler, train_cfg.scheduler.type)(
            getattr(torch.optim, train_cfg.optimizer.type)(
                model.parameters(),
                **train_cfg.optimizer.params
            ),
            **train_cfg.scheduler.params
        ),
        epochs=train_cfg.epochs,
        alpha=train_cfg.alpha,
        patience=train_cfg.patience
    )

    out_put_dir = HydraConfig.get().runtime.output_dir  # 모델 저장할 때 사용

    # 테스트 시작
    test_loss, covered_loss = test(
        model=model,
        test_loader=test_loader,
        device=device,
        save_root=out_put_dir,
        max_demand=dataset.cluster_dataset.max_demand,
        coverage=dataset.node_dataset.coverage
    )

    model_path = f"{out_put_dir}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved to {model_path}")

    metric = {  # multi run시 결과 저장용
        # "train_loss": train_losses[-1],
        # "val_loss": val_losses[-1],
        # "test_loss": test_loss,
        "covered_loss": covered_loss,
        # "epochs": config.train.epochs
    }
    results.append(metric)


if __name__ == "__main__":
    run()
    log.info(results)
