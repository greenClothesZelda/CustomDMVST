import torch
import torch.nn.functional as F
from torch.utils.data import Subset

import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from pathlib import Path

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from models.GNVSTNet import *
from models.loader.DemandDataset import MyDataset
from omegaconf import OmegaConf

from runners.test import test_loop

log = logging.getLogger(__name__)
results = []  # multi run시 결과 한눈에 보기 위해 사용

def set_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    # 결정론적 연산을 위한 설정 (필요 시)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

@hydra.main(config_path="configs", version_base=None)
def run(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    log.info(f"Using device: {device}")

    output_dir = HydraConfig.get().runtime.output_dir

    # 데이터셋 및 데이터로더 설정
    dataset = MyDataset(
        root=Path(config.dataset.root),
        time_step=config.dataset.time_step,
    )
    len_dataset = len(dataset)
    IR_dataset_size = int(len_dataset * config.split.ir_ratio)
    Train_dataset_size = int(len_dataset * config.split.train_ratio)

    indices = list(range(len_dataset))
    
    IR_dataset = Subset(dataset, indices[:IR_dataset_size])
    Train_dataset = Subset(dataset, indices[IR_dataset_size:IR_dataset_size + Train_dataset_size])
    Test_dataset = Subset(dataset, indices[IR_dataset_size + Train_dataset_size:])
    log.info(f"Dataset sizes - IR: {len(IR_dataset)}, Train: {len(Train_dataset)}, Test: {len(Test_dataset)}")
    
    ir_module = IRModule(IR_dataset)
    model = IRVSTNet(ir_module=ir_module, **config.model['IRVSTNet'])
    trainer_model = ModelTrainer(model).to(device)

    args = TrainingArguments(
        **config['train'],
        output_dir=output_dir,
        report_to=[],
        log_level='info'
    )

    trainer = Trainer(
        model=trainer_model,
        args=args,
        train_dataset=Train_dataset,
        eval_dataset=Test_dataset,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(**config.callbacks.early_stopping)]
    )
    trainer.train()
    
    test_results = test_loop(trainer_model, Test_dataset, output_dir, device)
    results.append(test_results)

if __name__ == "__main__":
    run()
    log.info(results)
