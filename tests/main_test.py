import pathlib
from datetime import datetime

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from pytest_ml_tdd_example.models import Net
from pytest_ml_tdd_example.training import test, train


@pytest.fixture
def exp_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    timestamp = datetime.today().strftime(r"%Y%m%d-%H%M%S")
    exp_dir = tmp_path / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


@pytest.fixture
def transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


@pytest.fixture
def train_dataset(transform: transforms.Compose) -> torch.utils.data.Dataset:
    return datasets.MNIST("./data", train=True, download=True, transform=transform)


@pytest.fixture
def test_dataset(transform: transforms.Compose) -> torch.utils.data.Dataset:
    return datasets.MNIST("./data", train=False, download=True, transform=transform)


@pytest.fixture
def train_batch_size() -> int:
    return 64


@pytest.fixture
def test_batch_size() -> int:
    return 64


@pytest.fixture
def train_loader(
    train_dataset: torch.utils.data.Dataset,
    train_batch_size: int,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )


@pytest.fixture
def test_loader(
    test_dataset: torch.utils.data.Dataset,
    test_batch_size: int,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )


@pytest.fixture
def model(device: torch.device) -> nn.Module:
    model = Net()
    model = model.to(device)
    return model


@pytest.fixture
def learning_rate() -> float:
    return 1.0


@pytest.fixture
def optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return optim.Adadelta(model.parameters(), lr=learning_rate)


@pytest.fixture
def scheduler(
    optimizer: torch.optim.Optimizer, step_size: int = 1, gamma: float = 0.7
) -> torch.optim.lr_scheduler.LRScheduler:
    return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)


@pytest.mark.parametrize(
    argnames="epochs",
    argvalues=(
        1,
        # 10, # Too long for test, so skip in actual test run
    ),
)
def test_main(
    exp_dir: pathlib.Path,
    epochs: int,
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    log_interval: int = 10,
    dry_run: bool = False,
    save_model: bool = True,
):
    for epoch in range(1, epochs + 1):
        train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            dry_run=dry_run,
        )
        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), exp_dir / "mnist_cnn.pt")
