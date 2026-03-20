import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import Subset
from transformers import Trainer, TrainingArguments

from models.GNVSTNet import IRModule, IRVSTNet, ModelTrainer, collate_fn
from models.loader.DemandDataset import MyDataset
from runners.test import test_loop

log = logging.getLogger(__name__)
results = []


def set_seed(seed):
    import numpy as np
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(labels, tuple):
        labels = labels[0]
    predictions = np.asarray(predictions, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)

    mae = np.mean(np.abs(predictions - labels))
    mape = np.mean(np.abs(predictions - labels) / (labels + 1.0)) * 100.0
    return {
        'mae': float(mae),
        'mape': float(mape)
    }

import torch.nn as nn
class DMVSTLoss(nn.Module):
    def __init__(self, lambda_rel=1.0, reduction="mean"):
        super().__init__()
        self.lambda_rel = lambda_rel
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        diff = y_true - y_pred
        abs_diff = torch.abs(diff)
        loss = abs_diff / (1.0 + y_true) + self.lambda_rel * abs_diff
        if self.reduction == "sum":
            loss = loss.sum()
        return loss.mean()


@hydra.main(config_path="configs", version_base=None)
def run(config):
    set_seed(config.seed)
    device = torch.device(config.device)
    log.info(f"Using device: {device}")

    output_dir = HydraConfig.get().runtime.output_dir

    dataset = MyDataset(
        root=Path(config.dataset.root),
        time_step=config.dataset.time_step,
        num_nodes=config.dataset.num_nodes,
        size=config.dataset.size
    )
    len_dataset = len(dataset)
    train_end = int(len_dataset * config.split.train_ratio)
    warmup = config.model.IRModule.k

    if train_end <= warmup:
        raise ValueError(f"train_end ({train_end}) must be greater than warmup ({warmup}).")
    if train_end >= len_dataset:
        raise ValueError(f"train_end ({train_end}) must be smaller than dataset length ({len_dataset}).")

    train_indices = list(range(warmup, train_end))
    test_indices = list(range(train_end, len_dataset))

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    log.info(
        "Dataset sizes - RetrievalOnly: %s, Train: %s, Test: %s",
        warmup,
        len(train_dataset),
        len(test_dataset)
    )

    ir_module = IRModule(dataset, device, k=config.model.IRModule.k)
    model = IRVSTNet(ir_module=ir_module, **config.model['IRVSTNet'])
    trainer_model = ModelTrainer(model, loss=DMVSTLoss()).to(device)

    args = TrainingArguments(
        **config['train'],
        output_dir=output_dir,
        report_to=[],
        log_level='info'
    )

    trainer = Trainer(
        model=trainer_model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train()

    test_results = test_loop(trainer_model, test_dataset, output_dir, device)
    results.append(test_results)


if __name__ == "__main__":
    run()
    log.info(results)
