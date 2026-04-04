import torch

from main import DMVSTLoss
from models.GNVSTNet import ModelTrainer
from models.STResNet import STResNet


def test_stresnet_forward_and_backward():
    model = STResNet(
        H=2,
        W=3,
        external_dim=12,
        nb_flow=1,
        len_c=3,
        len_p=1,
        len_t=1,
        nb_filter=8,
        nb_residual_unit=2,
        use_external=True,
        external_hidden=16,
    )

    demands_series = torch.randn(4, 5, 2, 3, requires_grad=True)
    weather = torch.randn(4, 4)
    time = torch.randn(4, 8)
    sample_idx = torch.arange(4)

    output = model(demands_series, weather, time, sample_idx)
    assert output.shape == (4, 6)
    output.sum().backward()

    grad_tensors = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grad_tensors)


def test_model_trainer_keeps_loss_and_relu_predictions():
    model = STResNet(
        H=2,
        W=3,
        external_dim=12,
        nb_flow=1,
        len_c=3,
        len_p=1,
        len_t=1,
        nb_filter=8,
        nb_residual_unit=1,
        use_external=True,
        external_hidden=16,
    )
    trainer_model = ModelTrainer(model, loss=DMVSTLoss(lambda_rel=1.0))

    batch_size = 2
    outputs = trainer_model(
        demands_series=torch.randn(batch_size, 5, 2, 3),
        weather=torch.randn(batch_size, 4),
        time=torch.randn(batch_size, 8),
        sample_idx=torch.arange(batch_size),
        labels=torch.ones(batch_size, 6),
        return_dict=True,
    )

    assert outputs["predictions"].shape == (batch_size, 6)
    assert outputs["loss"].ndim == 0
    assert torch.all(outputs["predictions"] >= 0)
