import torch
import hydra
from hydra.core.hydra_config import HydraConfig
import logging

log = logging.getLogger(__name__)
results = [] #multi run시 결과 한눈에 보기 위해 사용

@hydra.main(config_path="configs", version_base=None)
def run(config):
    device = torch.device(config.device)
    log.info(f"Using device: {device}")
    dataset_cfg = config.dataset #모델에 입력하는 방법은 unpacking(**)사용
    model_cfg = config.model
    train_cfg = config.train

    log.debug(f"data_cfg={dataset_cfg}")
    log.info(f"model_cfg={model_cfg}")
    log.info(f"train_cfg={train_cfg}")

    out_put_dir = HydraConfig.get().runtime.output_dir #모델 저장할 때 사용
    log.info(f"output dir:{out_put_dir}")
    metric = { #multi run시 결과 저장용
        "accuracy" : 0.95,
        "loss" : 0.02,
        "epochs":config.train.epochs
    }
    results.append(metric)

if __name__ == "__main__":
    run()
    log.info(results)