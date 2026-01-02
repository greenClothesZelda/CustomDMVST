import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import logging

from models.GNVSTNet import GNVSTNet
from models.loader.GNVSTDataset import MyGraphDataset
from models.loader.TemporalDataset import TemporalDataset
from torch_geometric.loader import DataLoader
from runners.trainer import *

log = logging.getLogger(__name__)
results = [] #multi run시 결과 한눈에 보기 위해 사용

def dmvst_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    weights = torch.full_like(y_true, 1.65).to(y_true.device)
    weights[y_true <= 0.01] = 1.65
    weights[(y_true > 0.08) & (y_true <= 0.1)] = 3.986
    weights[(y_true > 0.16) & (y_true <= 0.18)] = 10.690
    weights[y_true > 0.18] = 30.490

    # 3. 요소별 MAE 계산 (원래 코드에서 y -> y_true, y_pred로 수정)
    element_wise_loss = torch.abs(y_pred - y_true)

    # 4. 가중치 적용
    weighted_loss = element_wise_loss * weights

    # 5. 배치 손실 계산 (평균)
    return torch.mean(weighted_loss)

@hydra.main(config_path="configs", version_base=None)
def run(config):
    device = torch.device(config.device)
    log.info(f"Using device: {device}")
    dataset_cfg = config.dataset #모델에 입력하는 방법은 unpacking(**)사용
    model_cfg = config.model
    train_cfg = config.train

    # 데이터셋 및 데이터로더 설정
    dataset = MyGraphDataset(**dataset_cfg)
    temporal_dataset = TemporalDataset(root="data/raw/meteorological_data.csv", time_step=dataset_cfg.time_step)

    train_size = int(len(dataset) * config.loader.train.ratio)
    val_size = int(len(dataset) * config.loader.val.ratio)
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]


    train_loader = DataLoader(train_dataset, batch_size=config.loader.train.batch_size, shuffle=config.loader.train.shuffle)  
    val_loader = DataLoader(val_dataset, batch_size=config.loader.val.batch_size, shuffle=config.loader.val.shuffle)  
    test_loader = DataLoader(test_dataset, batch_size=config.loader.test.batch_size, shuffle=config.loader.test.shuffle)
    #로더 설정 끝

    # 모델 초기화
    model = GNVSTNet(
        context_dim=0,
        temporal_data_size=model_cfg.temporal_data_size,
        GNN_configs=model_cfg.GNN,
        LSTM_configs=model_cfg.LSTM,
    ).to(device)

    # 훈련 시작
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        temporal_dataset=None,
        device=device,optimizer=getattr(torch.optim, train_cfg.optimizer.type)(model.parameters(), **train_cfg.optimizer.params),
        
        criterion=dmvst_loss,
        scheduler=getattr(torch.optim.lr_scheduler, train_cfg.scheduler.type)(
            getattr(torch.optim, train_cfg.optimizer.type)(
                model.parameters(),
                **train_cfg.optimizer.params
                ),
              **train_cfg.scheduler.params
            ),
        epochs=train_cfg.epochs
    )

    out_put_dir = HydraConfig.get().runtime.output_dir #모델 저장할 때 사용

    # 테스트 시작
    accuracy = test(
        model=model,
        test_loader=test_loader,
        temporal_dataset=None,
        device=device,
        save_root=out_put_dir
    )
    
    model_path = f"{out_put_dir}/final_model.pth"
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved to {model_path}")

    metric = { #multi run시 결과 저장용
        "train_loss" : train_losses[-1],
        "val_loss" : val_losses[-1],
        "accuracy" : accuracy,
        "epochs":config.train.epochs
    }
    results.append(metric)

if __name__ == "__main__":
    run()
    log.info(results)